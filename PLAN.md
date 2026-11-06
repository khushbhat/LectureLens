# LectureLens — Build Plan

Multimodal (speech + slide) summarization of educational videos, free-tier only.

## Phases (each verified by actually running it)

- [x] **P0 Scaffolding** — layout, `config.yaml`, `requirements.txt`, `.env.example`, `.gitignore`, venv
- [x] **P1 Core infra** — config loader, quota manager (SQLite ledger, token-bucket throttle, backoff, PT-midnight reset), LLM response cache, providers (Gemini / Groq / Ollama / Mock / Extractive-none) + router with fallback, model-availability check
- [x] **P2 Ingest → transcript** — yt-dlp (≤480p video, audio, subs, chapters), FFmpeg audio (16 kHz WAV + low-bitrate Opus/MP3, ≤10-min chunks w/ overlap), caption cleaning (rolling lines, tags, [Music]), Groq Whisper, local faster-whisper, overlap de-dup + timestamp offsets, transcript.json/.txt/.srt
- [x] **P3 Preprocess** — fillers, repetition loops, sentence split w/ timestamps
- [x] **P4 Slides + OCR** — frame sampling, pHash + SSIM change detection, min duration, sharpest frame, EasyOCR, title detection, header/footer removal, near-dup merge, `visual_mode`
- [x] **P5 Alignment + batching** — slide-aligned segments, TextTiling fallback (MiniLM), merge/split, token-budgeted batches
- [x] **P6 Hierarchical summarization** — map/reduce (recursive), prompts in `prompts/`, pydantic validation + repair, single-pass & TextRank baselines
- [x] **P7 Verification** — retrieval + local NLI cross-encoder, faithfulness score
- [x] **P8 Export + CLI + resume** — MD / PDF (fpdf2) / JSON, `run`, `resume`, `quota`, `eval`, `run_log.json`
- [x] **P9 Streamlit app**
- [x] **P10 Evaluation** — 4 systems, ROUGE / BERTScore / concept coverage / slide-concept recall / faithfulness / cost / WER, quota-aware grid runner, tables (md + tex), figures, human-eval form, report.md
- [x] **P11 Tests** — mocked provider, no network
- [x] **P12 Docs** — README (setup, keys, free tier, methodology), docs/architecture.md
- [ ] **P13 End-to-end runs** — short slide lecture, ~60-min lecture, no-slide video, eval with placeholder refs

## Real API usage log
| Date | Purpose | Provider/model | Calls |
|---|---|---|---|
| 2026-11-06 | all development + tests | mock (offline) | 0 real |

## Status notes
- P1–P12 built and verified offline: mock end-to-end run on `samples/synthetic_gd.mp4`, mock eval grid, 42 pytest tests (no network).
- P13 waiting on `.env` keys. Local stages (download / transcript / slides / OCR) for the 3 YouTube test videos run without keys.
- Test videos: neso_process (7 min, slides), mit_rnn (57 min, slides), carroll_entropy (6 min, talking head).
