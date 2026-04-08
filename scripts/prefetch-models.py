#!/usr/bin/env python3
"""Pre-fetch WhisperX models into the local cache.

Run during ansible deploy so the first real mimir startup doesn't pause
for model downloads. Idempotent — second run is a no-op (models cached
in ~/.cache/huggingface/hub).

If MIMIR_MODEL fails to load (e.g. unsupported by faster-whisper), the
script exits non-zero and the deploy fails — caller should retry with
a different model name.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    model_name = os.environ.get("MIMIR_MODEL", "distil-large-v3")
    compute_type = os.environ.get("MIMIR_COMPUTE_TYPE", "int8")
    language = os.environ.get("MIMIR_LANGUAGE", "en")

    print(f"prefetch: importing whisperx ...", flush=True)
    import whisperx  # noqa: E402

    vad_method = os.environ.get("MIMIR_VAD_METHOD", "silero")
    print(
        f"prefetch: loading whisper model {model_name!r} "
        f"(compute_type={compute_type}, vad={vad_method}) ...",
        flush=True,
    )
    try:
        model = whisperx.load_model(
            model_name,
            device="cpu",
            compute_type=compute_type,
            vad_method=vad_method,
        )
    except Exception as e:
        # Note: WhisperX defaults to pyannote VAD which is HF-gated. We
        # always pass vad_method="silero" (free, no token) to avoid that
        # gotcha. If you see "Could not import module 'Pipeline'" here,
        # check that vad_method is actually being passed through.
        print(f"prefetch: FAILED to load model {model_name!r}: {e}", file=sys.stderr)
        return 2
    print(f"prefetch: ok — {type(model).__name__}", flush=True)

    print(f"prefetch: loading wav2vec2 alignment model for language {language!r} ...", flush=True)
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language, device="cpu")
    except Exception as e:
        print(f"prefetch: FAILED to load alignment model for {language!r}: {e}", file=sys.stderr)
        # Non-fatal: mimir v0 doesn't actually use alignment. Warn and continue.
        print("prefetch: continuing without alignment model", file=sys.stderr)
    else:
        print(f"prefetch: ok — alignment model loaded", flush=True)

    print("prefetch: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
