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
    MIMIR_MODEL             WhisperX model name (default distil-large-v3)
    MIMIR_LANGUAGE          ISO 639-1 language code (default en)
    MIMIR_COMPUTE_TYPE      ctranslate2 compute type (default int8)
    MIMIR_BATCH_SIZE        WhisperX batch size (default 16)
    MIMIR_WINDOW_SECONDS    audio window per inference (default 20)
    MIMIR_VAD_METHOD        whisperx VAD method (default silero)
    MIMIR_TCP_HOST          fanout TCP bind host (default 0.0.0.0)
    MIMIR_TCP_PORT          fanout TCP bind port (default 7200)

Architecture: N+2 threads.

  reader_thread(label)   one per source. Blocks on heimdall recv,
                         appends to that source's bytearray buffer under
                         a shared condition variable; trims to bound
                         memory; notifies the worker when new bytes
                         arrive.
  worker_thread          single worker. Iterates sources in rotating
                         order, picks whichever has a full window ready,
                         extracts WINDOW_SECONDS of audio, runs
                         WhisperX, emits each segment as a line prefixed
                         with `[LABEL]`.
  tcp_accept_thread      accepts TCP clients, adds to a subscriber list.

Emit() fans out the line to all subscribers under the same lock pattern
as heimdall's audio fanout. Slow subscribers are dropped.

Speaker diarization (diart on the `[MEETING]` source) is planned as a
follow-up commit — see docs/decisions/0008-mimir-source-tagging-and-diart.md
in the loki repo. This module previously ran pyannote-per-window diarization
that was silently broken in production and gave labels with no cross-window
identity; it has been removed.
"""

from __future__ import annotations

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
WINDOW_BYTES = SAMPLE_RATE * WINDOW_SECONDS * 2  # 2 bytes per sample
MAX_BUFFER_BYTES = SAMPLE_RATE * 60 * 2  # cap each source at 60 s of audio


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
                with audio_buffer_cond:
                    buf = audio_buffers[label]
                    buf.extend(chunk)
                    if len(buf) > MAX_BUFFER_BYTES:
                        drop = len(buf) - MAX_BUFFER_BYTES
                        del buf[:drop]
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
            # Advance rotation so the other sources get first crack next
            # iteration. (label_index[ready] + 1) wraps naturally via %.
            rotation_start = (label_index[ready] + 1) % len(labels)

        label = ready  # for clarity in logs + output below

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

        # Emit raw segment text prefixed with the source label. The
        # transcript is meant to flow as continuous prose; downstream
        # consumers (odin, nc, the host's Claude Code via loki-meeting)
        # use the `[LABEL]` prefix as the *only* speaker attribution —
        # there is no per-window timestamp. The journal still gets
        # stamped by systemd-journald, so debugging timing isn't lost.
        for seg in nonempty:
            text = (seg.get("text") or "").strip()
            line = f"[{label}] {text}"
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
                conn.send(
                    f"# mimir transcribe stream — model={MODEL_NAME} "
                    f"window={WINDOW_SECONDS}s sources=[{sources_desc}]\n".encode()
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
    log.info(
        "mimir starting: model=%s lang=%s window=%ds sources=[%s] tcp=%s:%d",
        MODEL_NAME, LANGUAGE, WINDOW_SECONDS, sources_desc, TCP_HOST, TCP_PORT,
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

    # One reader thread per source, plus one worker and one TCP accept.
    threads: list[threading.Thread] = []
    for label, socket_path in SOURCES.items():
        threads.append(threading.Thread(
            target=reader_thread,
            args=(label, socket_path),
            name=f"audio-reader[{label}]",
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
