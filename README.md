# mimir

Transcribe module for [loki](https://github.com/3loc/loki). Subscribes to heimdall's audio socket, runs WhisperX on CPU, and streams transcribed text lines over TCP.

Named after Mímir, the Norse god of wisdom — Odin gave an eye to drink from his well.

## What it does

Reads 16 kHz mono PCM from a heimdall Unix socket, transcribes 20-second windows with WhisperX (faster-whisper + int8 + Silero VAD, CPU-only), emits each non-empty segment as a line on a TCP fanout server.

## Connect

```
nc agneta 7200
```

You should see lines like:

```
[10:42:03] alright let's get started with the security review
[10:42:18] the first item is the SAML config we discussed last week
```

## Install

Requires Arch Linux, `uv` (installed by the playbook), and a running `heimdall@meeting.service` to consume audio from.

```
make deploy
```

First deploy is slow — downloads ~2 GB of model files and torch wheels. Subsequent deploys are fast.

## Status

v0 — single audio source, no diarization, no Claude integration. See `CLAUDE.md`.

## License

TBD
