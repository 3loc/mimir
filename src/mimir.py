#!/usr/bin/env python3
"""mimir — transcribe daemon for loki.

Subscribes to a heimdall audio Unix socket, runs WhisperX on rolling
windows, and emits one transcript line per non-empty segment to all
clients connected to a TCP fanout server.

Configuration via environment variables (set by systemd EnvironmentFile):

    MIMIR_HEIMDALL_SOCKET   Unix socket to subscribe to
                            (default /run/heimdall/meeting.sock)
    MIMIR_MODEL             WhisperX model name (default distil-large-v3)
    MIMIR_LANGUAGE          ISO 639-1 language code (default en)
    MIMIR_COMPUTE_TYPE      ctranslate2 compute type (default int8)
    MIMIR_BATCH_SIZE        WhisperX batch size (default 16)
    MIMIR_WINDOW_SECONDS    audio window per inference (default 20)
    MIMIR_TCP_HOST          fanout TCP bind host (default 0.0.0.0)
    MIMIR_TCP_PORT          fanout TCP bind port (default 7200)

Architecture: three threads.

  reader_thread       blocks on heimdall recv, appends to a bytearray
                      buffer under a lock; trims to bound memory.
  worker_thread       extracts WINDOW_SECONDS of audio at a time, runs
                      WhisperX, emits each segment as a line.
  tcp_accept_thread   accepts TCP clients, adds to a subscriber list.

Emit() fans out the line to all subscribers under the same lock pattern
as heimdall's audio fanout. Slow subscribers are dropped.
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

HEIMDALL_SOCKET = os.environ.get("MIMIR_HEIMDALL_SOCKET", "/run/heimdall/meeting.sock")
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

# Speaker diarization via pyannote. Off by default — set MIMIR_DIARIZE=1
# to turn it on, and provide HF_TOKEN via /etc/mimir/hf-token.env.
# Caveat: speaker labels (SPEAKER_00, SPEAKER_01, …) are LOCAL to each
# call. SPEAKER_00 in one window is not necessarily the same physical
# person as SPEAKER_00 in the next window — pyannote doesn't maintain
# identity across separate diarize() calls. For real cross-window
# identity you'd need to either accumulate longer audio chunks before
# diarizing, or use a second mic source tagged at the device level.
DIARIZE_ENABLED = os.environ.get("MIMIR_DIARIZE", "0") == "1"
DIARIZE_MODEL = os.environ.get(
    "MIMIR_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

SAMPLE_RATE = 16000  # heimdall produces 16 kHz mono s16le
WINDOW_BYTES = SAMPLE_RATE * WINDOW_SECONDS * 2  # 2 bytes per sample
MAX_BUFFER_BYTES = SAMPLE_RATE * 60 * 2  # cap at 60 s of audio


# ─── logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("mimir")


# ─── shared state ────────────────────────────────────────────────────────────

shutdown_event = threading.Event()

audio_buffer = bytearray()
audio_buffer_lock = threading.Lock()
audio_buffer_cond = threading.Condition(audio_buffer_lock)

tcp_subscribers: list[socket.socket] = []
tcp_subscribers_lock = threading.Lock()


# ─── audio reader (heimdall → buffer) ────────────────────────────────────────

def reader_thread() -> None:
    """Connect to heimdall and append PCM bytes to the shared buffer.

    Reconnects on socket loss with exponential backoff up to 5 s. Trims
    the buffer from the front when it exceeds MAX_BUFFER_BYTES, keeping
    the most recent audio (drops oldest, on the assumption that stale
    audio is less interesting than fresh).
    """
    backoff = 0.5
    while not shutdown_event.is_set():
        try:
            log.info("audio: connecting to %s", HEIMDALL_SOCKET)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(HEIMDALL_SOCKET)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            log.warning("audio: heimdall not ready (%s); retrying in %.1fs", e, backoff)
            shutdown_event.wait(backoff)
            backoff = min(backoff * 2, 5.0)
            continue

        log.info("audio: connected")
        backoff = 0.5
        try:
            while not shutdown_event.is_set():
                chunk = sock.recv(8192)
                if not chunk:
                    log.warning("audio: heimdall closed the connection")
                    break
                with audio_buffer_cond:
                    audio_buffer.extend(chunk)
                    if len(audio_buffer) > MAX_BUFFER_BYTES:
                        drop = len(audio_buffer) - MAX_BUFFER_BYTES
                        del audio_buffer[:drop]
                        log.warning("audio: trimmed %d stale bytes (worker too slow?)", drop)
                    audio_buffer_cond.notify_all()
        except OSError as e:
            log.error("audio: read error: %s", e)
        finally:
            try:
                sock.close()
            except OSError:
                pass

        if not shutdown_event.is_set():
            log.info("audio: reconnecting in %.1fs", backoff)
            shutdown_event.wait(backoff)
            backoff = min(backoff * 2, 5.0)

    log.info("audio: reader exiting")


# ─── worker (buffer → WhisperX → emit) ───────────────────────────────────────

def _assign_speaker(seg: dict, diarize_segments) -> str | None:
    """Pick the speaker label whose time range overlaps the segment most.

    `diarize_segments` is whatever pyannote's DiarizationPipeline returned
    via WhisperX — currently a pandas.DataFrame with columns 'start',
    'end', 'speaker'. We iterate, find the diarization row whose
    [start, end] overlaps the transcript segment's [start, end] the most,
    and return its 'speaker' value. None if no overlap.
    """
    s_start = float(seg.get("start") or 0.0)
    s_end = float(seg.get("end") or s_start)
    best_overlap = 0.0
    best_speaker: str | None = None
    try:
        for _, row in diarize_segments.iterrows():
            d_start = float(row.get("start", 0.0))
            d_end = float(row.get("end", 0.0))
            overlap = max(0.0, min(s_end, d_end) - max(s_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(row.get("speaker") or "")
    except Exception:
        # Pandas API differences / unexpected shape — fall back gracefully.
        return None
    return best_speaker or None


def worker_thread(model, diarize_pipeline) -> None:
    """Pull WINDOW_BYTES at a time, transcribe, optionally diarize, emit."""
    import numpy as np

    while not shutdown_event.is_set():
        # Wait until enough audio is available.
        with audio_buffer_cond:
            while len(audio_buffer) < WINDOW_BYTES and not shutdown_event.is_set():
                audio_buffer_cond.wait(timeout=1.0)
            if shutdown_event.is_set():
                break
            window_bytes = bytes(audio_buffer[:WINDOW_BYTES])
            del audio_buffer[:WINDOW_BYTES]

        # Convert int16 → float32 in [-1, 1] (whisperx wants float32 mono).
        audio = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        t0 = time.monotonic()
        try:
            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
        except Exception:
            log.exception("transcribe: failed on %ds window", WINDOW_SECONDS)
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
                "transcribe: %ds window in %.2fs (rtf=%.2fx) → %d segment(s)",
                WINDOW_SECONDS, dt, rtf, len(nonempty),
            )
        else:
            log.debug(
                "transcribe: %ds window in %.2fs (rtf=%.2fx) → silent",
                WINDOW_SECONDS, dt, rtf,
            )

        # Optional: run pyannote diarization on the same audio window so
        # each segment can be tagged with a speaker label.
        diarize_segments = None
        if diarize_pipeline is not None and nonempty:
            try:
                diarize_segments = diarize_pipeline(audio)
            except Exception:
                log.exception("diarize: failed on %ds window", WINDOW_SECONDS)
                diarize_segments = None

        # Emit raw segment text — no timestamp prefix. The transcript is
        # meant to flow as continuous prose; downstream consumers (odin,
        # nc, anything else) join lines and present them to humans
        # without per-window timestamps. The journal still gets stamped
        # by systemd-journald, so debugging timing isn't lost.
        for seg in nonempty:
            text = (seg.get("text") or "").strip()
            speaker: str | None = None
            if diarize_segments is not None:
                speaker = _assign_speaker(seg, diarize_segments)
            if speaker:
                line = f"[{speaker}] {text}"
            else:
                line = text
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
                conn.send(
                    f"# mimir transcribe stream — model={MODEL_NAME} window={WINDOW_SECONDS}s\n".encode()
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

    log.info(
        "mimir starting: model=%s lang=%s window=%ds heimdall=%s tcp=%s:%d",
        MODEL_NAME, LANGUAGE, WINDOW_SECONDS, HEIMDALL_SOCKET, TCP_HOST, TCP_PORT,
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

    diarize_pipeline = None
    if DIARIZE_ENABLED:
        if not HF_TOKEN:
            log.error(
                "diarization: MIMIR_DIARIZE=1 but HF_TOKEN is not set in the "
                "environment (expected via /etc/mimir/hf-token.env). "
                "Continuing WITHOUT diarization."
            )
        else:
            log.info("loading pyannote diarization model %s ...", DIARIZE_MODEL)
            t1 = time.monotonic()
            try:
                from whisperx.diarize import DiarizationPipeline  # noqa: E402
                diarize_pipeline = DiarizationPipeline(
                    model_name=DIARIZE_MODEL,
                    use_auth_token=HF_TOKEN,
                    device="cpu",
                )
                log.info("loaded diarization in %.1fs", time.monotonic() - t1)
            except Exception:
                log.exception(
                    "diarization: failed to load pipeline. Continuing WITHOUT "
                    "diarization. Check that you accepted the EULA at "
                    "https://huggingface.co/%s and that HF_TOKEN is valid.",
                    DIARIZE_MODEL,
                )
                diarize_pipeline = None

    threads = [
        threading.Thread(target=reader_thread, name="audio-reader", daemon=True),
        threading.Thread(
            target=worker_thread, args=(model, diarize_pipeline),
            name="worker", daemon=True,
        ),
        threading.Thread(target=tcp_accept_thread, name="tcp-accept", daemon=True),
    ]
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
