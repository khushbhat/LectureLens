# LectureLens architecture

## Pipeline

```mermaid
flowchart TD
    A[Video file / YouTube URL] --> B[Ingest<br/>yt-dlp: ≤480p video, audio, captions, chapters]
    B --> C{Human captions?}
    C -- yes --> D[Caption cleaning<br/>rolling-line de-dup, tags, noise]
    C -- no --> E[Audio: FFmpeg 16 kHz mono<br/>10-min chunks, 3 s overlap]
    E --> F[Groq Whisper large-v3-turbo<br/>free tier]
    F -- quota / error --> G[faster-whisper local<br/>CPU int8]
    F --> H[Merge chunks<br/>offsets + overlap de-dup]
    G --> H
    D --> I[Preprocess<br/>fillers, loops, sentences + timestamps]
    H --> I
    B --> J[Keyframes: 1 frame / 2 s<br/>pHash + SSIM + pixel-diff, min duration]
    J --> K[Local OCR EasyOCR<br/>lines, title, header/footer removal, reveal merge]
    K --> L{visual_mode}
    L -- high --> M[Slide-aligned segments<br/>slide + speech in its time window]
    L -- low --> N[Speech-only topic segments<br/>MiniLM TextTiling]
    I --> M
    I --> N
    M --> O[Quota-aware batching<br/>token budget per call]
    N --> O
    O --> P[MAP: 1 LLM call / batch<br/>per-segment JSON]
    P --> Q{fits reduce budget?}
    Q -- no --> R[Group reduce<br/>recursive]
    R --> Q
    Q -- yes --> S[REDUCE: 1 call<br/>summary, key points, notes, glossary, chapters, quiz]
    S --> T[Verify locally<br/>MiniLM retrieval + NLI cross-encoder]
    T --> U[Export: Markdown · PDF · JSON · Streamlit]
```

## LLM call path (every request)

```mermaid
flowchart LR
    R[request] --> C{LLM cache hit<br/>any provider?}
    C -- yes --> OUT[response, 0 quota]
    C -- no --> Q[Quota manager<br/>daily caps · RPM/TPM throttle]
    Q -- daily cap reached --> FB[next provider]
    Q -- ok --> CALL[provider call]
    CALL -- 429 short / 5xx --> BO[backoff + jitter<br/>honour retry-after] --> Q
    CALL -- 429 per-day --> MARK[mark exhausted today] --> FB
    CALL -- ok --> SAVE[cache + ledger] --> OUT
    FB -- none left --> PAUSE[save state · exit<br/>lecturelens resume run_id]
```

Provider chain: Gemini 3.5 Flash-Lite (free) → Groq `openai/gpt-oss-120b` (free) → Ollama (local) → extractive TextRank (no LLM).

## Run folder (`runs/<slug>_<hash>/`)

| File | Stage |
|---|---|
| `source.json`, `media/` | ingest |
| `transcript.json/.txt/.srt`, `asr/chunk_*.json` | transcript (per-chunk cache) |
| `sentences.json` | preprocess |
| `keyframes.json`, `keyframes/slide_*.png`, `thumb_*.jpg` | slide detection |
| `ocr_raw.json`, `slides.json` | OCR |
| `segments.json` / `segments_speech.json` | alignment / segmentation |
| `map/batches.json`, `map/batch_*.json`, `map/usage.jsonl` | map calls |
| `reduce/level*_group*.json`, `reduce/final.json`, `reduce/usage.jsonl` | reduce calls |
| `lecture.md`, `lecture.pdf`, `lecture.json` | export |
| `state.json`, `run_log.json` | status, timings, calls, tokens, config snapshot |
