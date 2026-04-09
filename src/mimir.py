#!/usr/bin/env python3
"""mimir — transcribe daemon for loki.

Subscribes to one or more heimdall audio Unix sockets, runs WhisperX on
rolling windows per source, and emits each transcript segment tagged
with its source label (e.g. `[MEETING]`, `[TED]`) to all clients
connected to a TCP fanout server.

Configuration via environment variables (set by systemd EnvironmentFile):

    MIMIR_SOURCES           comma-separated LABEL=PATH pairs of heimdall
                            sockets to subscribe to. Example:
                              MEETING=/run/heimdall/meeting.sock,TED=/run/heimdall/ted.sock
                            (default: the two entries above)
    MIMIR_DIARIZE_SOURCES   comma-separated LABEL=SOCKET pairs of mimir-diart
                            sidecar event sockets. Each labeled source
                            gets its [LABEL] transcript lines tagged
                            with [LABEL-SPKn] sub-speaker labels from
                            diart. Example:
                              MEETING=/run/mimir/diart-meeting.sock
                            Default: empty (no diarization). If diart
                            events are unavailable for a segment, mimir
                            falls back to the plain [LABEL] prefix
                            (fail-open, not fail-closed).
    MIMIR_MODEL             WhisperX model name (default distil-large-v3)
    MIMIR_LANGUAGE          ISO 639-1 language code (default en)
    MIMIR_COMPUTE_TYPE      ctranslate2 compute type (default int8)
    MIMIR_BATCH_SIZE        WhisperX batch size (default 16)
    MIMIR_WINDOW_SECONDS    audio window per inference (default 20)
    MIMIR_VAD_METHOD        whisperx VAD method (default silero)
    MIMIR_TCP_HOST          fanout TCP bind host (default 0.0.0.0)
    MIMIR_TCP_PORT          fanout TCP bind port (default 7200)

Architecture: N+M+2 threads.

  reader_thread(label)       one per heimdall source. Blocks on heimdall
                             recv, appends to that source's bytearray
                             buffer under a shared condition variable;
                             trims to bound memory; notifies the worker
                             when new bytes arrive. Tracks `head_wall_time`
                             per source — the wall-clock time of the byte
                             currently at the buffer head — so the worker
                             can correlate extracted windows with diart
                             speaker events that use wall-clock anchors.

  diart_reader_thread(label) one per MIMIR_DIARIZE_SOURCES entry.
                             Subscribes to a mimir-diart sidecar's Unix
                             socket, reads newline-delimited JSON events
                             ({"event": "start", "audio_start_wall": ...}
                             and {"event": "track", "audio_start": ...,
                             "audio_end": ..., "speaker": "speaker0"}),
                             and appends them to a per-source rolling
                             event list under diart_events_lock.

  worker_thread              single worker. Iterates sources in rotating
                             order, picks whichever has a full window
                             ready, extracts WINDOW_SECONDS of audio,
                             runs WhisperX, and for each segment queries
                             the diart event list for the source (if
                             any) to find the overlapping speaker. Emits
                             `[LABEL-SPKn] text` if a speaker is found,
                             `[LABEL] text` otherwise (fail-open).

  tcp_accept_thread          accepts TCP clients, adds to a subscriber list.

Emit() fans out the line to all subscribers under the same lock pattern
as heimdall's audio fanout. Slow subscribers are dropped.

Speaker diarization lives in a separate mimir-diart sidecar process
with its own venv (diart's pinned deps are incompatible with whisperx).
Failure isolation: if diart is down, mimir keeps transcribing without
speaker labels. See docs/decisions/0008-mimir-source-tagging-and-diart.md
in the loki repo for the rationale, and mimir_diart.py for the sidecar.

This module previously ran in-process pyannote-per-window diarization
that was silently broken in production and gave labels with no
cross-window identity; that code has been removed entirely.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime


# ─── config ──────────────────────────────────────────────────────────────────

def _parse_sources(env_value: str) -> dict[str, str]:
    """Parse MIMIR_SOURCES as LABEL=PATH[,LABEL=PATH...].

    Preserves insertion order (Python 3.7+ dict ordering) so the
    worker's rotation order is deterministic from the env string.
    Raises ValueError on malformed input rather than silently
    dropping entries — we'd rather crash loud at startup than
    quietly ignore a misconfigured source.
    """
    result: dict[str, str] = {}
    for pair in env_value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"MIMIR_SOURCES entry missing '=': {pair!r}")
        label, path = pair.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"MIMIR_SOURCES entry has empty label or path: {pair!r}")
        if label in result:
            raise ValueError(f"MIMIR_SOURCES has duplicate label {label!r}")
        result[label] = path
    if not result:
        raise ValueError("MIMIR_SOURCES is empty — at least one LABEL=PATH required")
    return result


# Two-source default — MEETING (Elgato HDMI via heimdall@meeting) and
# TED (close-talking lavalier via heimdall@ted). Override in the systemd
# EnvironmentFile if you need a different shape (e.g. just MEETING for
# headless-agneta mode, or adding a third source for in-person room mic).
SOURCES = _parse_sources(os.environ.get(
    "MIMIR_SOURCES",
    "MEETING=/run/heimdall/meeting.sock,TED=/run/heimdall/ted.sock",
))

# Optional mimir-diart sidecar event sockets. Empty by default — the
# sidecar is opt-in and must be deployed separately (see
# systemd/mimir-diart.service and src/mimir_diart.py). When set, each
# labeled source's [LABEL] transcript lines get tagged with
# [LABEL-SPKn] sub-speaker labels whenever a diart event overlaps the
# segment. If the sidecar is down, mimir emits plain [LABEL] and keeps
# going — fail-open.
DIARIZE_SOURCES: dict[str, str] = {}
_diarize_env = os.environ.get("MIMIR_DIARIZE_SOURCES", "").strip()
if _diarize_env:
    DIARIZE_SOURCES = _parse_sources(_diarize_env)
    # Every diarize label must correspond to an existing source label;
    # otherwise we'd be correlating speaker events against a stream we
    # don't even transcribe.
    unknown = set(DIARIZE_SOURCES) - set(SOURCES)
    if unknown:
        raise ValueError(
            f"MIMIR_DIARIZE_SOURCES refers to unknown labels {sorted(unknown)}; "
            f"every entry must match a MIMIR_SOURCES label"
        )

MODEL_NAME = os.environ.get("MIMIR_MODEL", "distil-large-v3")
LANGUAGE = os.environ.get("MIMIR_LANGUAGE", "en")
COMPUTE_TYPE = os.environ.get("MIMIR_COMPUTE_TYPE", "int8")
BATCH_SIZE = int(os.environ.get("MIMIR_BATCH_SIZE", "16"))
WINDOW_SECONDS = int(os.environ.get("MIMIR_WINDOW_SECONDS", "20"))
TCP_HOST = os.environ.get("MIMIR_TCP_HOST", "0.0.0.0")
TCP_PORT = int(os.environ.get("MIMIR_TCP_PORT", "7200"))
# WhisperX defaults to pyannote VAD which is HF-gated and would require
# a HuggingFace token. Silero is open and works fine for meeting audio.
VAD_METHOD = os.environ.get("MIMIR_VAD_METHOD", "silero")

SAMPLE_RATE = 16000  # heimdall produces 16 kHz mono s16le
BYTES_PER_SEC = SAMPLE_RATE * 2  # 2 bytes per sample
WINDOW_BYTES = BYTES_PER_SEC * WINDOW_SECONDS
MAX_BUFFER_BYTES = BYTES_PER_SEC * 60  # cap each source at 60 s of audio

# How much diart event history to keep in memory per source, in whole
# events, before trimming. Diart emits tracks on every chunk (~0.5s)
# and the per-chunk annotation contains multiple running tracks, so
# the event stream is noisy. 20000 entries ≈ a few minutes of audio;
# trim to 10000 when we hit the ceiling.
DIART_EVENTS_CEILING = 20000
DIART_EVENTS_FLOOR = 10000


# ─── logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("mimir")


# ─── shared state ────────────────────────────────────────────────────────────

shutdown_event = threading.Event()

# One bytearray per source, keyed by label. All buffers share a single
# condition variable so the worker can sleep on "any source has new
# bytes" without a per-source cond var / busy-poll loop.
audio_buffers: dict[str, bytearray] = {label: bytearray() for label in SOURCES}
audio_buffer_lock = threading.Lock()
audio_buffer_cond = threading.Condition(audio_buffer_lock)

# Wall-clock time (unix epoch seconds) of the byte currently at the
# head of each audio_buffer. None when the buffer is empty / no bytes
# have ever been received. Updated by reader_thread on first fill and
# on trim; by worker_thread on window extract. Used to map transcript
# segments to wall-clock times for correlation with diart events.
head_wall_times: dict[str, float | None] = {label: None for label in SOURCES}

# Per-source rolling list of (audio_start, audio_end, speaker) tuples
# received from the mimir-diart sidecar. audio_start/audio_end are in
# seconds relative to that source's audio_start_wall anchor (stored
# separately below). Appended by diart_reader_thread; read by
# worker_thread under diart_events_lock.
diart_events: dict[str, list[tuple[float, float, str]]] = {
    label: [] for label in DIARIZE_SOURCES
}

# Per-source audio anchor — the unix-epoch wall-clock time when the
# sidecar received its first audio byte. None until the sidecar emits
# its "start" event. Needed to convert diart's audio-relative times to
# wall clock for correlation with mimir's own (also wall-clock-tracked)
# audio buffers.
diart_audio_start_wall: dict[str, float | None] = {
    label: None for label in DIARIZE_SOURCES
}

diart_events_lock = threading.Lock()

tcp_subscribers: list[socket.socket] = []
tcp_subscribers_lock = threading.Lock()


# ─── audio reader (heimdall → per-source buffer) ─────────────────────────────

def reader_thread(label: str, socket_path: str) -> None:
    """Connect to one heimdall source and append PCM bytes to audio_buffers[label].

    Reconnects on socket loss with exponential backoff up to 5 s. Trims
    the per-source buffer from the front when it exceeds MAX_BUFFER_BYTES,
    keeping the most recent audio (drops oldest, on the assumption that
    stale audio is less interesting than fresh). Each source has its
    own reader thread and its own buffer, but all readers share the
    global audio_buffer_cond so the worker thread can wake on any
    source's progress.
    """
    backoff = 0.5
    prefix = f"audio[{label}]"
    while not shutdown_event.is_set():
        try:
            log.info("%s: connecting to %s", prefix, socket_path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            log.warning("%s: heimdall not ready (%s); retrying in %.1fs",
                        prefix, e, backoff)
            shutdown_event.wait(backoff)
            backoff = min(backoff * 2, 5.0)
            continue

        log.info("%s: connected", prefix)
        backoff = 0.5
        try:
            while not shutdown_event.is_set():
                chunk = sock.recv(8192)
                if not chunk:
                    log.warning("%s: heimdall closed the connection", prefix)
                    break
                recv_wall = time.time()
                with audio_buffer_cond:
                    buf = audio_buffers[label]
                    # If the buffer was empty (or has never seen bytes),
                    # anchor head_wall_time at the wall time this chunk
                    # arrived — less the time it represents. The chunk
                    # is `len(chunk)` bytes of PCM, which at 32 kB/s is
                    # `len(chunk) / BYTES_PER_SEC` seconds of audio. So
                    # the oldest byte in the chunk represents audio
                    # captured at roughly `recv_wall - (len(chunk) /
                    # BYTES_PER_SEC)`. For our correlation purposes
                    # sub-second precision is plenty.
                    if not buf:
                        head_wall_times[label] = recv_wall - (len(chunk) / BYTES_PER_SEC)
                    buf.extend(chunk)
                    if len(buf) > MAX_BUFFER_BYTES:
                        drop = len(buf) - MAX_BUFFER_BYTES
                        del buf[:drop]
                        # Advance head_wall_time by the audio duration
                        # of the dropped bytes, so it still points at
                        # the (new) head byte's wall-clock capture time.
                        if head_wall_times[label] is not None:
                            head_wall_times[label] += drop / BYTES_PER_SEC
                        log.warning("%s: trimmed %d stale bytes (worker too slow?)",
                                    prefix, drop)
                    audio_buffer_cond.notify_all()
        except OSError as e:
            log.error("%s: read error: %s", prefix, e)
        finally:
            try:
                sock.close()
            except OSError:
                pass

        if not shutdown_event.is_set():
            log.info("%s: reconnecting in %.1fs", prefix, backoff)
            shutdown_event.wait(backoff)
            backoff = min(backoff * 2, 5.0)

    log.info("%s: reader exiting", prefix)


# ─── diart event subscriber (sidecar → in-memory event list) ─────────────────

def diart_reader_thread(label: str, socket_path: str) -> None:
    """Subscribe to a mimir-diart sidecar and stream its events into memory.

    Handles two event types:

      {"event": "start", "audio_start_wall": <unix>}
          Announces the wall-clock time of the first audio sample the
          sidecar received. Stored in diart_audio_start_wall[label];
          used by _find_speaker_for_segment to map audio-relative
          track times to absolute wall clock for correlation.

      {"event": "track", "audio_start": <s>, "audio_end": <s>,
       "speaker": "speaker0"}
          A speaker turn. Appended verbatim to diart_events[label].
          The sidecar may emit the same track multiple times as
          diart's running prediction refines; we don't dedupe because
          the lookup path (iterate + find max overlap) is robust to
          duplicates and fast enough.

    The list is trimmed in-place when it crosses DIART_EVENTS_CEILING,
    dropping the oldest half. Reconnects on socket loss with
    exponential backoff.
    """
    backoff = 0.5
    prefix = f"diart[{label}]"
    events = diart_events[label]

    while not shutdown_event.is_set():
        try:
            log.info("%s: connecting to %s", prefix, socket_path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            log.warning("%s: sidecar not ready (%s); retrying in %.1fs",
                        prefix, e, backoff)
            shutdown_event.wait(backoff)
            backoff = min(backoff * 2, 5.0)
            continue

        log.info("%s: connected", prefix)
        backoff = 0.5
        buf = b""
        try:
            while not shutdown_event.is_set():
                chunk = sock.recv(4096)
                if not chunk:
                    log.warning("%s: sidecar closed the connection", prefix)
                    break
                buf += chunk
                # Process any complete JSON lines in the buffer.
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except Exception:
                        log.warning("%s: bad JSON line: %r", prefix, line[:120])
                        continue

                    ev_type = event.get("event")
                    if ev_type == "start":
                        try:
                            start_wall = float(event["audio_start_wall"])
                        except (KeyError, TypeError, ValueError):
                            log.warning("%s: bad start event: %r", prefix, event)
                            continue
                        diart_audio_start_wall[label] = start_wall
                        log.info("%s: anchored audio_start_wall=%f",
                                 prefix, start_wall)

                    elif ev_type == "track":
                        try:
                            audio_start = float(event["audio_start"])
                            audio_end = float(event["audio_end"])
                            speaker = str(event["speaker"])
                        except (KeyError, TypeError, ValueError):
                            log.warning("%s: bad track event: %r", prefix, event)
                            continue
                        with diart_events_lock:
                            events.append((audio_start, audio_end, speaker))
                            if len(events) > DIART_EVENTS_CEILING:
                                # Drop the oldest half, keeping the tail
                                # (which is the most recent speaker info
                                # we care about for emit-time correlation).
                                del events[: len(events) - DIART_EVENTS_FLOOR]
                                log.info(
                                    "%s: trimmed event list to %d entries",
                                    prefix, len(events),
                                )

                    # Silently ignore unknown event types — forward compat.
        except OSError as e:
            log.error("%s: read error: %s", prefix, e)
        finally:
            try:
                sock.close()
            except OSError:
                pass

        if not shutdown_event.is_set():
            log.info("%s: reconnecting in %.1fs", prefix, backoff)
            shutdown_event.wait(backoff)
            backoff = min(backoff * 2, 5.0)

    log.info("%s: reader exiting", prefix)


def _format_speaker_label(diart_speaker: str) -> str:
    """Convert diart's internal 'speakerN' string into a clean display tag.

    Diart emits speaker labels like 'speaker0', 'speaker1', 'speaker2'
    — all lower-case, no separator. Downstream consumers (odin →
    zerokb → Claude Code) are much easier to read when each speaker
    is tagged 'SPEAKER-0', 'SPEAKER-1' etc, matching the shape of a
    typical interview transcript. Anything that doesn't fit the
    'speaker<digits>' pattern is uppercased as-is, so custom diart
    configs or future label schemes still round-trip sensibly.
    """
    if diart_speaker.startswith("speaker") and diart_speaker[7:].isdigit():
        return f"SPEAKER-{diart_speaker[7:]}"
    return diart_speaker.upper()


def _find_speaker_for_segment(
    label: str,
    seg_wall_start: float,
    seg_wall_end: float,
) -> str | None:
    """Find the diart speaker with maximum overlap for a [seg_wall_start, seg_wall_end] range.

    Returns the speaker label (e.g. "speaker0") or None if no diart
    events overlap the segment — the caller falls back to plain [LABEL]
    attribution in that case. Iterates all events for the source; the
    list is bounded by DIART_EVENTS_CEILING so this is O(ceiling) per
    call, which is negligible compared to whisperx inference.
    """
    anchor = diart_audio_start_wall.get(label)
    if anchor is None:
        return None  # sidecar hasn't emitted its start event yet
    events = diart_events.get(label)
    if not events:
        return None

    best_ov = 0.0
    best_spk: str | None = None
    with diart_events_lock:
        # Snapshot under the lock so the list isn't mutated mid-iter.
        # Small copy for a bounded list — cheap.
        snapshot = list(events)

    for audio_start, audio_end, speaker in snapshot:
        ev_wall_start = anchor + audio_start
        ev_wall_end = anchor + audio_end
        overlap = max(
            0.0, min(seg_wall_end, ev_wall_end) - max(seg_wall_start, ev_wall_start)
        )
        if overlap > best_ov:
            best_ov = overlap
            best_spk = speaker
    return best_spk


# ─── worker (per-source buffers → WhisperX → emit) ───────────────────────────

def _pick_ready_source(rotation_start: int) -> str | None:
    """Return the label of the first source (starting at rotation_start,
    wrapping) with at least WINDOW_BYTES of audio buffered. None if no
    source is ready. Caller must hold audio_buffer_lock.
    """
    labels = list(audio_buffers.keys())
    n = len(labels)
    for offset in range(n):
        label = labels[(rotation_start + offset) % n]
        if len(audio_buffers[label]) >= WINDOW_BYTES:
            return label
    return None


def worker_thread(model) -> None:
    """Pull WINDOW_BYTES from whichever source is ready, transcribe, emit.

    Sources are served in rotating order so no source can be starved
    by a consistently-louder neighbor. When multiple sources are
    simultaneously ready, the rotation index advances past the one we
    just served, giving the other source(s) first crack next round.
    """
    import numpy as np

    labels = list(audio_buffers.keys())
    label_index = {label: i for i, label in enumerate(labels)}
    rotation_start = 0

    while not shutdown_event.is_set():
        # Wait until at least one source has a full window.
        with audio_buffer_cond:
            ready = _pick_ready_source(rotation_start)
            while ready is None and not shutdown_event.is_set():
                audio_buffer_cond.wait(timeout=1.0)
                ready = _pick_ready_source(rotation_start)
            if shutdown_event.is_set():
                break
            # Extract the window from the ready source's buffer.
            window_bytes = bytes(audio_buffers[ready][:WINDOW_BYTES])
            del audio_buffers[ready][:WINDOW_BYTES]
            # Snapshot the wall-clock time of the window's FIRST byte
            # (which is head_wall_times[ready] BEFORE we advance it for
            # this extraction). Used later to correlate transcript
            # segments with diart speaker events.
            window_start_wall = head_wall_times[ready]
            if head_wall_times[ready] is not None:
                head_wall_times[ready] += WINDOW_BYTES / BYTES_PER_SEC
            # Advance rotation so the other sources get first crack next
            # iteration. (label_index[ready] + 1) wraps naturally via %.
            rotation_start = (label_index[ready] + 1) % len(labels)

        label = ready  # for clarity in logs + output below
        # Fallback if head_wall_time was somehow None (shouldn't happen
        # once reader has received any bytes, but be defensive): pretend
        # the window starts "now" so correlation is at least roughly OK.
        if window_start_wall is None:
            window_start_wall = time.time() - WINDOW_SECONDS

        # Convert int16 → float32 in [-1, 1] (whisperx wants float32 mono).
        audio = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        t0 = time.monotonic()
        try:
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
        except Exception:
            log.exception("transcribe[%s]: failed on %ds window", label, WINDOW_SECONDS)
            continue
        dt = time.monotonic() - t0
        rtf = dt / WINDOW_SECONDS

        segments = result.get("segments", []) if isinstance(result, dict) else []
        nonempty = [s for s in segments if (s.get("text") or "").strip()]
        # Only log INFO for windows that produced actual transcription.
        # Silent windows are the common case during quiet stretches and
        # would otherwise spam the journal at one INFO line per 5
        # seconds. They drop to DEBUG so debugging the worker is still
        # possible if you crank journald to debug level.
        if nonempty:
            log.info(
                "transcribe[%s]: %ds window in %.2fs (rtf=%.2fx) → %d segment(s)",
                label, WINDOW_SECONDS, dt, rtf, len(nonempty),
            )
        else:
            log.debug(
                "transcribe[%s]: %ds window in %.2fs (rtf=%.2fx) → silent",
                label, WINDOW_SECONDS, dt, rtf,
            )

        # Emit each segment. For diarized sources (listed in
        # MIMIR_DIARIZE_SOURCES), tag with `[SPEAKER-N]` where N is
        # the diart speaker index, or `[SPEAKER-?]` when diart hasn't
        # caught up to this window yet (diart's streaming pipeline
        # has roughly 20s of emit latency on the current agneta CPU,
        # so the first few segments after a source starts are
        # unavoidably unknown). For non-diarized sources, tag with
        # `[LABEL]` (e.g. `[TED]` — the label IS the speaker, by
        # hardware, because the source is a close-talking single-
        # speaker mic). The wall-clock range for each segment is
        # computed from the window's start wall plus the segment's
        # intra-window offset (seg.start / seg.end are seconds
        # relative to the window, which is anchored at
        # window_start_wall).
        for seg in nonempty:
            text = (seg.get("text") or "").strip()
            tag: str
            if label in DIARIZE_SOURCES:
                seg_rel_start = float(seg.get("start") or 0.0)
                seg_rel_end = float(seg.get("end") or seg_rel_start)
                seg_wall_start = window_start_wall + seg_rel_start
                seg_wall_end = window_start_wall + seg_rel_end
                diart_speaker = _find_speaker_for_segment(
                    label, seg_wall_start, seg_wall_end
                )
                if diart_speaker:
                    tag = _format_speaker_label(diart_speaker)
                else:
                    tag = "SPEAKER-?"
            else:
                tag = label
            line = f"[{tag}] {text}"
            print(line, flush=True)
            _broadcast(line + "\n")

    log.info("worker: exiting")


# ─── TCP fanout server ───────────────────────────────────────────────────────

def _broadcast(text: str) -> None:
    """Send a text line (already newline-terminated) to all subscribers.

    Subscribers are non-blocking sockets (set in tcp_accept_thread on
    accept). A healthy peer drains the kernel send buffer fast enough
    that send() returns the full byte count. A dead peer stuck in
    CLOSE-WAIT (e.g. ``nc`` Ctrl-C'd) eventually fills its send
    buffer; send() then raises BlockingIOError, at which point we
    drop it. This is the fix for the bug where six dead nc clients
    in CLOSE-WAIT froze the entire transcribe pipeline because
    sendall() was blocking forever on a dead socket while holding
    the broadcast lock.
    """
    data = text.encode("utf-8")
    dead: list[socket.socket] = []
    with tcp_subscribers_lock:
        for sub in tcp_subscribers:
            try:
                sent = sub.send(data)
            except BlockingIOError:
                log.warning("tcp: subscriber too slow (kernel buffer full), dropping")
                dead.append(sub)
                continue
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                log.info("tcp: subscriber dead (%s)", e)
                dead.append(sub)
                continue
            if sent < len(data):
                log.warning("tcp: subscriber partial write (%d/%d), dropping",
                            sent, len(data))
                dead.append(sub)
        for sub in dead:
            tcp_subscribers.remove(sub)
            try:
                sub.close()
            except OSError:
                pass


def tcp_accept_thread() -> None:
    """Accept TCP clients and add them to the subscriber list."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(8)
    server.settimeout(1.0)
    log.info("tcp: listening on %s:%d", TCP_HOST, TCP_PORT)

    try:
        while not shutdown_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            # Non-blocking sends are critical: a CLOSE-WAIT'd peer (e.g.
            # an nc client the user Ctrl-C'd) would otherwise block
            # sendall() forever once its kernel send buffer fills, and
            # freeze the transcribe pipeline. Same fix as heimdall.
            conn.setblocking(False)
            # Welcome banner — best-effort, ignore failures.
            try:
                sources_desc = ",".join(SOURCES.keys())
                diarize_desc = ",".join(DIARIZE_SOURCES.keys()) or "none"
                conn.send(
                    f"# mimir transcribe stream — model={MODEL_NAME} "
                    f"window={WINDOW_SECONDS}s sources=[{sources_desc}] "
                    f"diarize=[{diarize_desc}]\n".encode()
                )
            except (BlockingIOError, OSError):
                pass
            with tcp_subscribers_lock:
                tcp_subscribers.append(conn)
            log.info("tcp: subscriber from %s:%d connected (%d total)",
                     addr[0], addr[1], len(tcp_subscribers))
    finally:
        server.close()
        log.info("tcp: accept loop exiting")


# ─── main ────────────────────────────────────────────────────────────────────

def shutdown(signum, frame) -> None:
    log.info("received signal %d, shutting down", signum)
    shutdown_event.set()
    with audio_buffer_cond:
        audio_buffer_cond.notify_all()


def main() -> int:
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Silence WhisperX's per-window WARNING when the audio contains no
    # speech ("whisperx.vads.silero - WARNING - No active speech found
    # in audio"). It fires every 5 seconds during meeting silence and
    # generates ~1500 garbage log lines per hour of quiet. Real silero
    # errors (ERROR / CRITICAL) still propagate.
    logging.getLogger("whisperx.vads.silero").setLevel(logging.ERROR)

    sources_desc = ", ".join(f"{label}={path}" for label, path in SOURCES.items())
    if DIARIZE_SOURCES:
        diarize_desc = ", ".join(
            f"{label}={path}" for label, path in DIARIZE_SOURCES.items()
        )
    else:
        diarize_desc = "(none)"
    log.info(
        "mimir starting: model=%s lang=%s window=%ds sources=[%s] diarize=[%s] tcp=%s:%d",
        MODEL_NAME, LANGUAGE, WINDOW_SECONDS, sources_desc, diarize_desc,
        TCP_HOST, TCP_PORT,
    )

    log.info(
        "loading whisperx (%s, compute_type=%s, vad=%s) ...",
        MODEL_NAME, COMPUTE_TYPE, VAD_METHOD,
    )
    t0 = time.monotonic()
    import whisperx  # noqa: E402  (heavy import; intentional after logging)
    model = whisperx.load_model(
        MODEL_NAME,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        vad_method=VAD_METHOD,
    )
    log.info("loaded whisperx in %.1fs", time.monotonic() - t0)

    # One reader thread per source, one diart reader thread per
    # diarized source, plus one worker and one TCP accept.
    threads: list[threading.Thread] = []
    for label, socket_path in SOURCES.items():
        threads.append(threading.Thread(
            target=reader_thread,
            args=(label, socket_path),
            name=f"audio-reader[{label}]",
            daemon=True,
        ))
    for label, socket_path in DIARIZE_SOURCES.items():
        threads.append(threading.Thread(
            target=diart_reader_thread,
            args=(label, socket_path),
            name=f"diart-reader[{label}]",
            daemon=True,
        ))
    threads.append(threading.Thread(
        target=worker_thread, args=(model,),
        name="worker", daemon=True,
    ))
    threads.append(threading.Thread(
        target=tcp_accept_thread,
        name="tcp-accept", daemon=True,
    ))
    for t in threads:
        t.start()

    # Watchdog: if any critical thread dies unexpectedly, exit so systemd
    # restarts us.
    while not shutdown_event.is_set():
        for t in threads:
            if not t.is_alive():
                log.error("thread %s died unexpectedly, exiting", t.name)
                shutdown_event.set()
                break
        shutdown_event.wait(2.0)

    log.info("mimir stopped")
    # Use os._exit instead of returning normally so Python skips
    # module-level cleanup. libtorch / ctranslate2 / faster-whisper have
    # a long C++ destructor chain that segfaults / std::terminates if
    # called from the interpreter shutdown path, which would otherwise
    # leave the systemd unit in a "failed (core-dump)" state every time
    # we restart cleanly. systemd-journal is line-buffered so the
    # "mimir stopped" log line above is already flushed.
    os._exit(0)


if __name__ == "__main__":
    main()
