"""Stage 1 — Ingest a local video file or a YouTube URL.

YouTube: yt-dlp downloads a <=480p video-only stream (enough for slide OCR,
small on disk), the audio-only stream, and subtitles (manual preferred over
auto-generated). Records title, duration and chapters.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from .audio import probe_duration
from .utils import write_json

log = logging.getLogger(__name__)

URL_RE = re.compile(r"^https?://", re.I)


class IngestError(Exception):
    """Friendly, user-facing ingest failure."""


def is_url(source: str) -> bool:
    return bool(URL_RE.match(source.strip()))


def _friendly_ytdlp_error(msg: str) -> str:
    m = msg.lower()
    if "private video" in m:
        return "This YouTube video is private. Use a public/unlisted video or download it yourself and pass the file."
    if "sign in to confirm your age" in m or "age-restricted" in m or "inappropriate for some users" in m:
        return ("This video is age-restricted. Set ingest.cookies_from_browser (e.g. 'chrome') in config.yaml "
                "to use your logged-in browser session, or pass a local file.")
    if "sign in to confirm you" in m and "bot" in m:
        return ("YouTube is asking to confirm you're not a bot. Set ingest.cookies_from_browser in config.yaml, "
                "update yt-dlp (pip install -U yt-dlp), or download the video and pass the file.")
    if "live event" in m or "is live" in m or "premieres in" in m:
        return "This is a live stream / upcoming premiere. Wait until it has finished and been processed."
    if "video unavailable" in m or "not available" in m or "removed" in m:
        return "This video is unavailable (removed, region-blocked or the URL is wrong)."
    if "unsupported url" in m:
        return "Unsupported URL. Pass a YouTube video URL or a local video file path."
    return f"yt-dlp failed: {msg.strip()[:300]}"


def _ydl_opts(cfg, extra: dict) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True, "retries": 5}
    if cfg.get("ingest.cookies_from_browser"):
        opts["cookiesfrombrowser"] = (cfg.get("ingest.cookies_from_browser"),)
    opts.update(extra)
    return opts


def _pick_subs(info: dict, langs: list[str], allow_auto: bool) -> tuple[str | None, str | None]:
    """Return (lang, kind) preferring manual captions over auto-generated ones."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for lang in langs:
        for key in manual:
            if key == lang or key.startswith(lang + "-"):
                return key, "manual"
    if allow_auto:
        for lang in langs:
            for key in auto:
                if key == lang or key == f"{lang}-orig":
                    return key, "auto"
    return None, None


def fetch_youtube_info(url: str, cfg) -> dict:
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(_ydl_opts(cfg, {"skip_download": True})) as ydl:
            return ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise IngestError(_friendly_ytdlp_error(str(e))) from e


def ingest(source: str, run_dir: Path, cfg, info: dict | None = None) -> dict:
    """Download / locate media. Returns and writes `source.json`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    media = run_dir / "media"
    media.mkdir(exist_ok=True)

    if not is_url(source):
        p = Path(source).expanduser().resolve()
        if not p.exists():
            raise IngestError(f"File not found: {p}")
        meta = {
            "kind": "file", "source": str(p), "title": p.stem, "video_id": None,
            "video_path": str(p), "audio_path": str(p), "subtitle_path": None, "subtitle_kind": None,
            "duration": probe_duration(p), "chapters": [],
        }
        # sidecar subtitles next to a local file (lecture.srt / lecture.en.vtt)
        for ext in (".srt", ".vtt", ".en.srt", ".en.vtt"):
            side = p.with_name(p.stem + ext)
            if side.exists():
                meta["subtitle_path"], meta["subtitle_kind"] = str(side), "manual"
                break
        write_json(run_dir / "source.json", meta)
        return meta

    import yt_dlp

    info = info or fetch_youtube_info(source, cfg)
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise IngestError("This is a live stream / upcoming premiere; wait until it has ended and been processed.")
    if info.get("_type") == "playlist":
        raise IngestError("Playlists aren't supported — pass a single video URL.")

    vid = info.get("id")
    meta = {
        "kind": "youtube", "source": source, "video_id": vid, "title": info.get("title") or vid,
        "channel": info.get("channel") or info.get("uploader"),
        "duration": float(info.get("duration") or 0),
        "chapters": [{"title": c.get("title"), "start": c.get("start_time"), "end": c.get("end_time")}
                     for c in (info.get("chapters") or [])],
        "video_path": None, "audio_path": None, "subtitle_path": None, "subtitle_kind": None,
    }

    def run(opts: dict) -> None:
        try:
            with yt_dlp.YoutubeDL(_ydl_opts(cfg, opts)) as ydl:
                ydl.download([source])
        except yt_dlp.utils.DownloadError as e:
            raise IngestError(_friendly_ytdlp_error(str(e))) from e

    # subtitles (free, tiny): manual preferred; auto only if allowed
    lang, kind = _pick_subs(info, cfg.get("transcript.caption_langs", ["en"]),
                            bool(cfg.get("transcript.use_auto_captions", False)))
    if lang:
        run({"skip_download": True, "writesubtitles": kind == "manual", "writeautomaticsub": kind == "auto",
             "subtitleslangs": [lang], "subtitlesformat": "vtt/srt/best",
             "outtmpl": str(media / "subs.%(ext)s")})
        subs = sorted(media.glob("subs*.vtt")) + sorted(media.glob("subs*.srt"))
        if subs:
            meta["subtitle_path"], meta["subtitle_kind"] = str(subs[0]), kind
            log.info("Downloaded %s captions (%s)", kind, lang)

    # audio-only stream (needed unless captions are used; cheap, keep for WER / fallback)
    if not list(media.glob("audio.*")):
        run({"format": "ba[ext=m4a]/ba/b", "outtmpl": str(media / "audio.%(ext)s")})
    a = [p for p in media.glob("audio.*") if not p.name.endswith(".part")]
    meta["audio_path"] = str(a[0]) if a else None

    # <=480p video-only stream for slide detection
    if cfg.get("ingest.download_video", True):
        h = int(cfg.get("ingest.max_height", 480))
        if not list(media.glob("video.*")):
            run({"format": f"bv*[height<={h}][ext=mp4]/bv*[height<={h}]/b[height<={h}]/wv*/w",
                 "outtmpl": str(media / "video.%(ext)s")})
        v = [p for p in media.glob("video.*") if not p.name.endswith(".part")]
        meta["video_path"] = str(v[0]) if v else None

    if not meta["audio_path"] and not meta["subtitle_path"]:
        raise IngestError("Could not download audio or captions for this video.")
    if not meta["duration"] and meta["audio_path"]:
        meta["duration"] = probe_duration(meta["audio_path"])
    write_json(run_dir / "source.json", meta)
    return meta


def cleanup_media(run_dir: Path) -> None:
    """Delete downloaded media (keeps all derived artifacts)."""
    shutil.rmtree(run_dir / "media", ignore_errors=True)
