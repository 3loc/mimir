# CLAUDE.md — mimir

## What this is

**mimir** is the transcribe module for [loki](https://github.com/3loc/loki). It subscribes to heimdall's audio Unix socket, runs the audio through WhisperX on agneta's CPU, and emits a stream of timestamped transcript lines over TCP.

Named after Mímir, the Norse god of wisdom and knowledge — Odin sacrificed an eye to drink from his well. Same job, smaller scale: turn raw audio into knowledge for the rest of the loki stack to consume.

## What it does

```
heimdall ─/run/heimdall/meeting.sock──► mimir ──tcp://agneta:7200──► consumers
                  (16 kHz mono PCM)              (text lines)
```

- **Subscribes** to a heimdall audio socket as a Unix-socket client
- **Buffers** raw 16 kHz mono PCM into a rolling window
- **Transcribes** each 20-second window with WhisperX on CPU (faster-whisper backend, int8 quantization, Silero VAD)
- **Emits** a line per non-empty segment to all connected TCP clients:
  ```
  [10:42:03] alright let's get started with the security review
  [10:42:18] the first item is the SAML config we discussed last week
  ```

No diarization, no word-level alignment, no Claude integration. Pure text stream for now. Future: subscribes to multiple heimdall sockets (`meeting.sock` + `ted.sock`), tags each line by source, and exposes structured chunks instead of plain text.

## Why TCP and not stdout

A TCP fanout server (instead of just stdout) is the future-facing interface — the orchestrator (odin) will subscribe to it to consume transcript chunks for the Claude session. Same shape as zerokb's `tcp:7070` and heimdall's `http:7100`. Connect from anywhere on the LAN with `nc agneta 7200` to watch live.

## Hardware / environment

- **Host:** `agneta` (Minisforum UM790 Pro), Arch Linux, kernel 6.19
- **Compute:** CPU only — Ryzen 9 7940HS, 8 cores Zen4. No GPU offload, no ROCm, no NPU.
- **Python:** 3.12 (pinned via uv), in a project venv at `repos/mimir/.venv`. Not system Python (Arch ships 3.14, which is too new for the WhisperX dep chain).
- **Audio source:** the heimdall@meeting unix socket at `/run/heimdall/meeting.sock`. Mimir cannot run without heimdall (the systemd unit `Requires=heimdall@meeting.service`).

## ADR

See [loki/docs/decisions/0007-transcribe-whisperx-small.md](../../docs/decisions/0007-transcribe-whisperx-small.md) for the engine choice rationale, the model-selection process, the diarization deferral, and the Python-3.12 pin.

## Status

- [x] Scaffolding + ansible playbook
- [x] WhisperX installs cleanly on agneta with `uv` + Python 3.12 + CPU torch
- [x] **First text stream end-to-end from heimdall** — verified live: real news broadcast audio captured through the Elgato, transcribed, streamed over `tcp://agneta:7200`, watched from the Mac Studio with `nc 192.168.10.13 7200`
- [x] First benchmark on real audio: **`distil-large-v3` int8 on agneta CPU**
  - 20-second window → transcribe in 4.3 s, **RTF 0.22×** (4.6× faster than realtime)
  - 5-second window → transcribe in 3.7 s, **RTF 0.74×** (~26% headroom — borderline if other CPU load shows up)
  - Per-call overhead is roughly fixed at ~3.5 s (model invocation + VAD + segmentation); only ~0.2 s scales with audio length
  - **Implication:** larger windows are far more efficient. 10–20 s is the sweet spot; 5 s works but leaves little CPU headroom for the rest of the loki stack
- [ ] systemd hardening (currently only `NoNewPrivileges=yes`)
- [ ] Multi-source support (`meeting` + `ted`) — when the USB mic arrives, mimir grows from one Unix socket subscriber into two and tags transcript lines by source
- [ ] Structured output (JSON instead of plain text lines) — when odin needs it
- [ ] Output deduplication for windowed audio (5-s windows occasionally cut a word at the boundary) — only worth fixing if it bothers a real consumer

## Discovered gotchas (recorded so we don't relitigate)

These are the actual install/runtime obstacles hit during the first mimir deploy. ADR 0007 anticipated some of these in the abstract; this section records the concrete forms they took on agneta.

1. **`numpy<2` is wrong for WhisperX 3.8+.** Newer WhisperX requires `numpy>=2.1`. Earlier ADR drafts said the opposite; corrected.
2. **`whisperx==3.8.2` and `3.8.3` are yanked from PyPI.** Word-timestamps bug (#1372) and faster-whisper incompatibility (#1385). uv handled this correctly once we removed the bad numpy pin. Pin to `>=3.8.4` explicitly.
3. **`torch`, `torchaudio`, AND `torchvision` must all come from the same index.** uv pulls torch/torchaudio from the explicit `pytorch-cpu` index but defaults `torchvision` to PyPI, which gives an incompatible build. Symptom: `RuntimeError: operator torchvision::nms does not exist`, then transformers can't load `Pipeline`, and you get a confusing `Could not import module 'Pipeline'` error from WhisperX. Fix: add `torchvision` to deps AND to `[tool.uv.sources]`.
4. **WhisperX defaults to pyannote VAD, which is HuggingFace-gated.** Even when you skip diarization, `whisperx.load_model()` tries to load pyannote's segmentation model, which requires a HF token + EULA. Pass `vad_method="silero"` to bypass entirely. Silero is open, fast, and works fine for meeting audio.
5. **`libtorchcodec` warning at startup is benign.** torchcodec wants ffmpeg 4–7 for video decoding; agneta has ffmpeg 8. WhisperX doesn't actually use torchcodec for audio loading, so the warning is just noise. Filtered out of `make probe` output.
6. **Heimdall's broadcast loop blocks on dead-but-still-ESTAB Unix-socket peers.** When a probe or earlier mimir is `kill -9`'d, the kernel doesn't immediately mark the socket as closed for the writing side. heimdall's `sendall()` then blocks indefinitely on a full kernel send buffer for that dead peer, freezing the entire fanout. Workaround: `systemctl restart heimdall@meeting.service` to clear stale subscribers. Real fix is in heimdall (non-blocking sends or per-subscriber send timeouts) — tracked in heimdall's CLAUDE.md.

## Layout

```
mimir/
├── CLAUDE.md
├── README.md
├── Makefile
├── pyproject.toml          uv project pinning Python 3.12 + WhisperX + CPU torch
├── .gitignore              ignores .venv/ and downloaded models
├── ansible/
│   ├── inventory.yml       agneta as localhost via ansible_connection: local
│   └── deploy-mimir.yml    install uv, sync venv, prefetch models, install service
├── src/
│   └── mimir.py            the daemon (the only Python file)
├── scripts/
│   └── prefetch-models.py  ansible runs this once during deploy
└── systemd/
    ├── mimir.service       long-running daemon, requires heimdall@meeting
    └── mimir.env           runtime config (HEIMDALL_SOCKET, MIMIR_MODEL, etc.)
```
