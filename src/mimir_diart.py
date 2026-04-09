#!/usr/bin/env python3
"""mimir-diart — streaming speaker diarization sidecar for mimir.

Sits alongside mimir on agneta. Subscribes to a heimdall audio Unix
socket (default /run/heimdall/meeting.sock), runs diart's
SpeakerDiarization streaming pipeline against the PCM byte stream,
and publishes speaker events as newline-delimited JSON on its own
Unix domain socket (default /run/mimir/diart-meeting.sock). mimir
subscribes to that socket alongside its existing heimdall sources
and uses the events to tag [MEETING] transcript segments with stable
per-speaker labels.

Why a sidecar?
    diart's pinned dep stack (torch 2.2.2, torchaudio 2.2.2,
    pyannote.audio 3.1.1, huggingface_hub 0.20.3) is incompatible
    with mimir's whisperx venv. Rather than force mimir onto a fragile
    downgrade, we run diart in its own venv and talk to it via an IPC
    socket. Failure isolation as a side benefit: diart can crash and
    mimir keeps transcribing — it just loses the `-SPKn` suffix until
    the sidecar recovers.

Event format (one JSON object per newline):

    {"event": "start", "audio_start_wall": <unix_epoch>, "label": "MEETING"}
        Emitted once, after the first audio byte is received. Announces
        the wall-clock time of the first sample. mimir uses this plus
        the per-track `audio_start`/`audio_end` offsets to compute
        absolute wall times for each speaker turn.

    {"event": "track", "audio_start": <float_s>, "audio_end": <float_s>,
     "speaker": "speaker0", "emit_wall": <unix_epoch>, "label": "MEETING"}
        Emitted on every hook fire — diart's running prediction may
        include the same (start, end, speaker) multiple times across
        hook fires as it refines its view. mimir deduplicates by the
        (audio_start, audio_end, speaker) triple.

Configuration via env (systemd EnvironmentFile):

    MIMIR_DIART_HEIMDALL_SOCKET  heimdall socket to subscribe to
                                 (default /run/heimdall/meeting.sock)
    MIMIR_DIART_OUTPUT_SOCKET    Unix socket to bind for mimir clients
                                 (default /run/mimir/diart-meeting.sock)
    MIMIR_DIART_SAMPLE_RATE      PCM sample rate (default 16000)
    MIMIR_DIART_BLOCK_DURATION   chunk duration in seconds (default 0.5)
    MIMIR_DIART_LABEL            label to embed in each event, matches
                                 mimir's MIMIR_SOURCES label for this
                                 source (default MEETING)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import AudioSource
from diart.inference import StreamingInference


# ─── config ──────────────────────────────────────────────────────────────────

HEIMDALL_SOCKET = os.environ.get(
    "MIMIR_DIART_HEIMDALL_SOCKET", "/run/heimdall/meeting.sock"
)
OUTPUT_SOCKET = os.environ.get(
    "MIMIR_DIART_OUTPUT_SOCKET", "/run/mimir/diart-meeting.sock"
)
SAMPLE_RATE = int(os.environ.get("MIMIR_DIART_SAMPLE_RATE", "16000"))
BLOCK_DURATION = float(os.environ.get("MIMIR_DIART_BLOCK_DURATION", "0.5"))
LABEL = os.environ.get("MIMIR_DIART_LABEL", "MEETING")


# ─── logging ─────────────────────────────────────────────────────────────────

# NOTE on force=True: pyannote.audio and pytorch-lightning both configure
# handlers on the root logger during import. By the time this module's
# top-level code runs, the root logger already has handlers attached, so
# a plain logging.basicConfig() call becomes a no-op and all our log.info()
# calls disappear into the void. force=True tells basicConfig to nuke any
# existing handlers and set up a fresh one, which makes the sidecar's own
# log lines actually reach stderr (and from there systemd-journald).
# Without this, you get a silent sidecar that looks alive but has no
# observable state and is impossible to debug when something goes wrong.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    force=True,
)
log = logging.getLogger("mimir-diart")
log.setLevel(logging.INFO)


# ─── shared state ────────────────────────────────────────────────────────────

shutdown_event = threading.Event()
subscribers: list[socket.socket] = []
subscribers_lock = threading.Lock()


# ─── heimdall → diart source ─────────────────────────────────────────────────

class HeimdallSocketAudioSource(AudioSource):
    """diart AudioSource backed by a heimdall Unix socket.

    Modeled on diart's MicrophoneAudioSource. A background reader
    thread connects to the heimdall Unix socket, converts incoming
    s16le bytes to float32 normalized to [-1, 1], reshapes to
    (channels=1, block_size), and pushes chunks onto a queue. The
    read() method (called by StreamingInference) drains the queue and
    emits chunks on self.stream.

    Records `self.audio_start_wall` — the wall clock time when the
    first audio byte was received — so the event hook can announce it
    downstream for mimir's time correlation.
    """

    def __init__(
        self,
        socket_path: str,
        sample_rate: int = 16000,
        block_duration: float = 0.5,
    ):
        super().__init__(uri=f"heimdall:{socket_path}", sample_rate=sample_rate)
        self.socket_path = socket_path
        self.block_size = int(round(block_duration * sample_rate))
        self.block_bytes = self.block_size * 2  # 2 bytes per s16le sample
        self._queue: queue.Queue = queue.Queue()
        self._is_closed = False
        self.audio_start_wall: float | None = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="heimdall-socket-reader",
        )

    def _reader_loop(self) -> None:
        """Blocking socket-read loop, chunks the stream into block_bytes frames.

        Reconnects on socket loss with a 1 s delay. Each assembled
        block is pushed onto self._queue as a (1, block_size) float32
        numpy array. The first frame sets audio_start_wall.
        """
        buf = bytearray()
        while not self._is_closed:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self.socket_path)
                log.info("heimdall: connected to %s", self.socket_path)
            except (FileNotFoundError, ConnectionRefusedError) as e:
                log.warning("heimdall: not ready (%s); retrying in 1s", e)
                time.sleep(1.0)
                continue

            try:
                while not self._is_closed:
                    chunk = sock.recv(self.block_bytes)
                    if not chunk:
                        log.warning("heimdall: connection closed")
                        break
                    buf.extend(chunk)
                    while len(buf) >= self.block_bytes:
                        if self.audio_start_wall is None:
                            self.audio_start_wall = time.time()
                            log.info(
                                "audio_start_wall=%f (first byte received)",
                                self.audio_start_wall,
                            )
                        frame = bytes(buf[: self.block_bytes])
                        del buf[: self.block_bytes]
                        samples = (
                            np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        waveform = samples.reshape(1, -1)  # (channels=1, block_size)
                        self._queue.put(waveform)
            except OSError as e:
                log.error("heimdall: read error: %s", e)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

            if not self._is_closed:
                time.sleep(0.5)

        log.info("heimdall reader exiting")

    def read(self) -> None:
        """diart's entry point — start the reader + drain the queue into self.stream."""
        self._reader_thread.start()
        while not self._is_closed:
            try:
                waveform = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.stream.on_next(waveform)
            except BaseException as e:
                log.exception("stream.on_next raised")
                self.stream.on_error(e)
                break
        self.stream.on_completed()

    def close(self) -> None:
        self._is_closed = True


# ─── output socket fanout ────────────────────────────────────────────────────

def publish_event(event: dict) -> None:
    """Send a JSON line to every subscriber; drop dead ones."""
    line = (json.dumps(event) + "\n").encode("utf-8")
    dead: list[socket.socket] = []
    with subscribers_lock:
        for sub in subscribers:
            try:
                sub.sendall(line)
            except (BrokenPipeError, ConnectionResetError, BlockingIOError, OSError) as e:
                log.info("subscriber dead (%s)", e)
                dead.append(sub)
        for sub in dead:
            subscribers.remove(sub)
            try:
                sub.close()
            except OSError:
                pass


def accept_loop(listen_sock: socket.socket) -> None:
    """Accept subscribers on the output Unix socket."""
    listen_sock.settimeout(1.0)
    while not shutdown_event.is_set():
        try:
            conn, _ = listen_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        with subscribers_lock:
            subscribers.append(conn)
        log.info("subscriber connected (%d total)", len(subscribers))
    log.info("accept loop exiting")


# ─── event hook (diart → publish) ────────────────────────────────────────────

def build_event_hook(source: HeimdallSocketAudioSource):
    """Return a closure that StreamingInference can call on every chunk."""
    announced_start = [False]

    def hook(result):
        # result is (Annotation, SlidingWindowFeature)
        try:
            annotation = result[0]
        except (TypeError, IndexError):
            log.warning("hook: unexpected result shape %r", type(result).__name__)
            return

        # Once the source has captured its first byte, announce the
        # audio_start_wall anchor so downstream consumers can map
        # audio-relative times back to wall clock.
        if source.audio_start_wall is not None and not announced_start[0]:
            publish_event({
                "event": "start",
                "audio_start_wall": source.audio_start_wall,
                "label": LABEL,
            })
            announced_start[0] = True
            log.info("announced audio_start_wall=%f", source.audio_start_wall)

        # Emit every track in the running prediction. Downstream (mimir)
        # deduplicates on (audio_start, audio_end, speaker).
        try:
            for segment, _, label in annotation.itertracks(yield_label=True):
                publish_event({
                    "event": "track",
                    "audio_start": float(segment.start),
                    "audio_end": float(segment.end),
                    "speaker": str(label),
                    "emit_wall": time.time(),
                    "label": LABEL,
                })
        except Exception:
            log.exception("hook: failed to iterate annotation")

    return hook


# ─── main ────────────────────────────────────────────────────────────────────

def shutdown(signum, frame) -> None:
    log.info("received signal %d, shutting down", signum)
    shutdown_event.set()


def main() -> int:
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    log.info(
        "mimir-diart starting: heimdall=%s output=%s label=%s sr=%d block=%.2fs",
        HEIMDALL_SOCKET, OUTPUT_SOCKET, LABEL, SAMPLE_RATE, BLOCK_DURATION,
    )

    # ─── output socket ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_SOCKET), exist_ok=True)
    try:
        os.unlink(OUTPUT_SOCKET)
    except FileNotFoundError:
        pass
    listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen_sock.bind(OUTPUT_SOCKET)
    listen_sock.listen(8)
    try:
        os.chmod(OUTPUT_SOCKET, 0o660)
    except OSError:
        pass
    log.info("listening on %s", OUTPUT_SOCKET)

    accept_thread = threading.Thread(
        target=accept_loop, args=(listen_sock,), daemon=True, name="accept",
    )
    accept_thread.start()

    # ─── diart pipeline ──────────────────────────────────────────────────────
    log.info("loading diart SpeakerDiarization pipeline ...")
    t0 = time.monotonic()
    pipeline = SpeakerDiarization(SpeakerDiarizationConfig(sample_rate=SAMPLE_RATE))
    log.info("loaded pipeline in %.1fs", time.monotonic() - t0)

    source = HeimdallSocketAudioSource(
        socket_path=HEIMDALL_SOCKET,
        sample_rate=SAMPLE_RATE,
        block_duration=BLOCK_DURATION,
    )

    inference = StreamingInference(
        pipeline,
        source,
        do_plot=False,
        show_progress=False,
        do_profile=False,
    )
    inference.attach_hooks(build_event_hook(source))

    log.info("starting inference loop (reads heimdall forever)")
    try:
        inference()
    except KeyboardInterrupt:
        pass
    except Exception:
        log.exception("inference crashed")
        return 1
    finally:
        source.close()
        try:
            listen_sock.close()
        except OSError:
            pass
        try:
            os.unlink(OUTPUT_SOCKET)
        except FileNotFoundError:
            pass

    log.info("mimir-diart stopped")
    # Use os._exit — same rationale as mimir.py: libtorch + pyannote have
    # a long C++ destructor chain that can segfault during interpreter
    # shutdown and leave the systemd unit in "failed (core-dump)".
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
