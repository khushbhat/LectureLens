"""Stage 3 — Transcript: YouTube captions -> Groq Whisper -> local faster-whisper.

Output segments `{id, start, end, text}` -> transcript.json / .txt / .srt,
plus the source used and the real-time factor.
"""

from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path

from .audio import merge_chunk_segments, probe_duration, split_audio, to_upload_mp3, to_wav16k
from .providers.base import ProviderUnavailable, RateLimitError, TransientError
from .quota import QuotaExhausted
from .utils import fmt_ts, read_json, srt_ts, write_json, write_text

log = logging.getLogger(__name__)


class NoSpeechError(Exception):
    pass


# ----------------------------------------------------------------- captions
_TS = r"(\d+:)?\d{1,2}:\d{2}[.,]\d{3}"
_CUE_RE = re.compile(rf"({_TS})\s*-->\s*({_TS})[^\n]*\n(.*?)(?=\n\s*\n|\Z)", re.S)
_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9.]*(?:\s[^>]*)?>|<\d{1,2}:\d{2}[^>]*>")
_NOISE_RE = re.compile(r"\[(?:music|applause|laughter|laughs|inaudible|silence|noise|background[^\]]*)\]|\((?:music|applause|laughter|laughs)\)|♪+", re.I)


def _ts_to_s(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def clean_caption_text(text: str) -> str:
    t = _TAG_RE.sub("", html.unescape(_TAG_RE.sub("", text)))
    t = _NOISE_RE.sub(" ", t)
    t = re.sub(r"^\s*>>\s*", "", t, flags=re.M)  # speaker-change markers
    t = re.sub(r"^\s*-\s+", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


def parse_captions(raw: str) -> list[dict]:
    """Parse WebVTT/SRT into clean segments.

    Handles YouTube "rolling" captions, where every cue repeats the previous
    line and appends a few new words: a line is emitted only once.
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    segs: list[dict] = []
    prev_lines: list[str] = []
    emitted_recent: list[str] = []
    for m in _CUE_RE.finditer(raw):
        start, end = _ts_to_s(m.group(1)), _ts_to_s(m.group(3))
        body = m.group(5)
        if end - start < 0.05:  # YouTube's 10 ms "snapshot" cues
            continue
        lines = [clean_caption_text(l) for l in body.split("\n")]
        lines = [l for l in lines if l and not re.fullmatch(r"\d+", l)]
        new = [l for l in lines if l not in prev_lines and l not in emitted_recent[-4:]]
        prev_lines = lines
        if not new:
            if segs and lines:
                segs[-1]["end"] = max(segs[-1]["end"], end)
            continue
        text = " ".join(new)
        emitted_recent.extend(new)
        segs.append({"start": start, "end": end, "text": text})
    # rolling captions: a cue's end overlaps the next cue; clamp
    for a, b in zip(segs, segs[1:]):
        if a["end"] > b["start"]:
            a["end"] = b["start"]
    out = [s for s in segs if s["text"]]
    for i, s in enumerate(out):
        s["id"] = i
    return out


# ---------------------------------------------------------------- ASR helpers
def _groq_chunk(whisper, piece: dict, quota, cfg, run_id: str | None) -> list[dict]:
    model = whisper.model
    attempt, max_retries = 0, int(cfg.get("quota.max_retries", 4))
    while True:
        quota.acquire("groq", model, audio_s=max(10.0, piece["duration"]), run_id=run_id)
        try:
            return whisper.transcribe(piece["path"])
        except RateLimitError as e:
            if e.daily:
                quota.mark_exhausted("groq", model, str(e))
                raise QuotaExhausted("groq", model, "daily audio quota exhausted (server)") from e
            if attempt >= max_retries or (e.retry_after or 0) > float(cfg.get("quota.max_wait_s", 120)):
                raise QuotaExhausted("groq", model, f"rate limited: {str(e)[:120]}") from e
            wait = (e.retry_after or 2 ** (attempt + 1)) + 1
            log.warning("Groq Whisper rate limited; waiting %.0fs", wait)
            if quota.on_wait:
                quota.on_wait(wait, "Groq Whisper rate limit")
            quota.sleeper(wait)
            attempt += 1
        except TransientError:
            if attempt >= max_retries:
                raise
            quota.sleeper(2 ** (attempt + 1))
            attempt += 1


_LOCAL_MODEL = {}


def _local_model(cfg):
    from faster_whisper import WhisperModel

    key = (cfg.get("transcript.local_model", "base"), cfg.get("transcript.local_compute_type", "int8"))
    if key not in _LOCAL_MODEL:
        log.info("Loading faster-whisper '%s' (%s, CPU) — first run downloads the model", *key)
        _LOCAL_MODEL[key] = WhisperModel(key[0], device="cpu", compute_type=key[1])
    return _LOCAL_MODEL[key]


def _local_chunk(path: str, cfg) -> list[dict]:
    model = _local_model(cfg)
    segments, _info = model.transcribe(path, language=cfg.get("transcript.language") or None,
                                       vad_filter=True, beam_size=1, condition_on_previous_text=False)
    return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]


# --------------------------------------------------------------------- main
def transcribe(run_dir: Path, source: dict, cfg, quota=None, run_id: str | None = None,
               progress=None) -> dict:
    """Produce transcript.json (cached). Chunk results are cached for resume."""
    out_json = run_dir / "transcript.json"
    cached = read_json(out_json)
    if cached:
        return cached

    t0 = time.time()
    duration = float(source.get("duration") or 0)
    order = list(cfg.get("transcript.order", ["captions", "groq", "local"]))
    result = None

    if "captions" in order and source.get("subtitle_path"):
        kind = source.get("subtitle_kind")
        if kind == "manual" or cfg.get("transcript.use_auto_captions", False):
            raw = Path(source["subtitle_path"]).read_text(encoding="utf-8", errors="replace")
            segs = parse_captions(raw)
            if segs:
                result = {"source": f"captions:{kind}", "model": None, "segments": segs}
                log.info("Transcript from %s captions: %d segments", kind, len(segs))

    if result is None:
        audio = source.get("audio_path")
        if not audio:
            raise NoSpeechError("No captions and no audio track available.")
        work = run_dir / "asr"
        work.mkdir(exist_ok=True)
        wav = to_wav16k(audio, work / "audio16k.wav")
        duration = duration or probe_duration(wav)

        use_groq = "groq" in order and cfg.api_key("GROQ_API_KEY")
        whisper = None
        if use_groq:
            from .providers.groq_llm import GroqWhisper

            whisper = GroqWhisper(cfg.get("transcript.groq_model", "whisper-large-v3-turbo"),
                                  cfg.api_key("GROQ_API_KEY"), language=cfg.get("transcript.language"))

        chunk_s = float(cfg.get("transcript.chunk_seconds", 600))
        overlap = float(cfg.get("transcript.chunk_overlap_s", 3))
        bitrate = cfg.get("transcript.groq_bitrate", "32k")
        if duration > float(cfg.get("transcript.chunk_if_longer_than_s", 1200)):
            pieces = split_audio(wav, work / "chunks", chunk_s, overlap, bitrate, duration)
        else:
            mp3 = to_upload_mp3(wav, work / "upload.mp3", bitrate)
            size_mb = mp3.stat().st_size / 1e6
            if size_mb > float(cfg.get("transcript.groq_max_upload_mb", 24)):
                pieces = split_audio(wav, work / "chunks", chunk_s, overlap, bitrate, duration)
            else:
                pieces = [{"index": 0, "path": str(mp3), "offset": 0.0, "duration": duration}]

        sources_used, chunk_results = set(), []
        for piece in pieces:
            cache_f = work / f"chunk_{piece['index']:03d}.json"
            got = read_json(cache_f)
            if got is None and whisper is not None:
                try:
                    segs = _groq_chunk(whisper, piece, quota, cfg, run_id)
                    got = {"source": "groq", "model": whisper.model, "segments": segs}
                except (QuotaExhausted, ProviderUnavailable, TransientError, RateLimitError) as e:
                    log.warning("Groq Whisper unavailable (%s) — using local faster-whisper", str(e)[:150])
                    whisper = None
                except Exception as e:  # unexpected API error: fall back, don't crash
                    log.warning("Groq Whisper failed (%s) — using local faster-whisper", str(e)[:150])
                    whisper = None
            if got is None:
                if "local" not in order:
                    raise RuntimeError("Groq transcription unavailable and local Whisper disabled in config")
                if progress:
                    progress(f"Local Whisper: chunk {piece['index'] + 1}/{len(pieces)}")
                got = {"source": "local", "model": f"faster-whisper-{cfg.get('transcript.local_model', 'base')}",
                       "segments": _local_chunk(piece["path"], cfg)}
            write_json(cache_f, got)
            sources_used.add(f"{got['source']}:{got['model']}")
            chunk_results.append((piece["offset"], got["segments"]))
            if progress:
                progress(f"Transcribed {piece['index'] + 1}/{len(pieces)} audio pieces")

        segs = merge_chunk_segments(chunk_results)
        result = {"source": "+".join(sorted(sources_used)), "model": None, "segments": segs,
                  "chunks": len(pieces)}

    if not result["segments"] or sum(len(s["text"]) for s in result["segments"]) < 20:
        raise NoSpeechError("No speech detected in this video (empty transcript).")

    elapsed = time.time() - t0
    duration = duration or result["segments"][-1]["end"]
    result.update({
        "duration": duration, "elapsed_s": round(elapsed, 2),
        "real_time_factor": round(elapsed / duration, 4) if duration else None,
        "language": cfg.get("transcript.language"),
    })
    write_json(out_json, result)
    write_text(run_dir / "transcript.txt",
               "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in result["segments"]))
    write_text(run_dir / "transcript.srt", "\n".join(
        f"{i + 1}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{s['text']}\n"
        for i, s in enumerate(result["segments"])))
    return result
