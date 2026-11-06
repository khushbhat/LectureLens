"""LectureLens — Streamlit app.  Run:  streamlit run app.py

Calls exactly the same library code as the CLI (lecturelens.pipeline).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from lecturelens.config import load_config
from lecturelens.export import match_slides_to_headings
from lecturelens.ingest import IngestError, is_url
from lecturelens.pipeline import Pipeline, QuotaPaused, RunOptions
from lecturelens.quota import QuotaManager
from lecturelens.utils import fmt_ts, read_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="LectureLens", page_icon="🎓", layout="wide")

BADGE = {"supported": "✅", "uncertain": "❔", "unsupported": "⚠️"}


@st.cache_resource
def get_cfg():
    return load_config()


cfg = get_cfg()
ss = st.session_state
ss.setdefault("result", None)
ss.setdefault("seek", 0)
ss.setdefault("quiz_checked", False)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎓 LectureLens")
    st.caption("Speech + slide summarization of lectures · free tier only")
    keys = cfg.keys_found()
    st.markdown("**API keys found** (values never shown)")
    st.markdown("\n".join(f"- {'✅' if v else '—'} `{k}`" for k, v in keys.items()))
    order = st.multiselect("LLM providers (in fallback order)", ["gemini", "groq", "ollama"],
                           default=[p for p in cfg.get("llm.order", []) if p in ("gemini", "groq", "ollama")])
    transcript_pref = st.radio("Transcript source", ["Captions → Groq → local", "Groq → local", "Local Whisper only"])
    use_slides = st.toggle("Slide detection + OCR", value=True)
    do_verify = st.toggle("Fact-check key points (local NLI)", value=True)

    st.markdown("**Today's free-tier usage**")
    q = QuotaManager(cfg)
    pairs = [("gemini", cfg.get("llm.gemini.model")), ("groq", cfg.get("llm.groq.model")), ("groq", cfg.get("transcript.groq_model"))]
    rows = q.report(pairs)
    st.dataframe(pd.DataFrame([{"model": f"{r['provider']}/{r['model']}", "used": r["requests_today"],
                                "left": "∞" if r["requests_left"] is None else r["requests_left"],
                                "blocked": "yes" if r["blocked"] else ""} for r in rows]),
                 hide_index=True, width="stretch")
    st.caption("Limits are in config.yaml → quota.limits. Check your real numbers in AI Studio / Groq console.")

    st.markdown("**Run history**")
    pipe0 = Pipeline(cfg)
    for r in pipe0.list_runs()[:12]:
        c1, c2 = st.columns([3, 1])
        icon = {"done": "✅", "paused": "⏸️", "error": "❌", "running": "⏳"}.get(r["status"], "•")
        c1.caption(f"{icon} {(r['title'] or r['run_id'])[:38]} · {r['duration']}")
        if r["status"] == "done" and c2.button("Open", key=f"open_{r['run_id']}"):
            ss.result = read_json(pipe0.runs_dir / r["run_id"] / "lecture.json")
            ss.seek = 0
        if r["status"] == "paused" and c2.button("Resume", key=f"res_{r['run_id']}"):
            ss.resume_id = r["run_id"]

transcript_order = {"Captions → Groq → local": ["captions", "groq", "local"], "Groq → local": ["groq", "local"],
                    "Local Whisper only": ["local"]}[transcript_pref]
options = RunOptions(use_slides=use_slides, providers=order or None, transcript_order=transcript_order, verify=do_verify)

# --------------------------------------------------------------------- input
st.header("Summarize a lecture")
c1, c2 = st.columns([3, 2])
url = c1.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
upload = c2.file_uploader("…or upload a video", type=["mp4", "mkv", "webm", "mov", "avi", "mp3", "wav", "m4a"])
source = None
if upload is not None:
    up_dir = Path("samples/uploads")
    up_dir.mkdir(parents=True, exist_ok=True)
    dst = up_dir / upload.name
    if not dst.exists() or dst.stat().st_size != upload.size:
        dst.write_bytes(upload.getbuffer())
    source = str(dst.resolve())
elif url.strip():
    source = url.strip()


@st.cache_data(show_spinner="Estimating quota needs…", ttl=600)
def preflight(src: str, use_slides_: bool, order_: tuple):
    return Pipeline(cfg).preflight(src, RunOptions(use_slides=use_slides_, providers=list(order_) or None))


if source:
    try:
        pf = preflight(source, use_slides, tuple(order))
        calls = next((r["calls"] for r in pf["providers"] if r["provider"] == pf["plan_provider"]), 0)
        toks = next((r["tokens"] for r in pf["providers"] if r["provider"] == pf["plan_provider"]), 0)
        st.info(f"**Pre-flight:** ~{calls} LLM calls, ~{toks:,} tokens, est. {max(1, pf['est_time_s'] // 60)} min "
                f"({fmt_ts(pf['duration'])} of video{', captions available' if pf.get('has_captions') else ''}).\n\n"
                + pf["message"].replace("\n", "  \n"))
    except IngestError as e:
        st.error(str(e))
        source = None
    except Exception as e:
        st.warning(f"Pre-flight failed: {e}")


def run_pipeline(src: str | None, resume_id: str | None = None):
    with st.status("Running LectureLens…", expanded=True) as status:
        wait_box = st.empty()

        def progress(msg: str):
            status.update(label=msg.strip("▶ "))
            if msg.startswith("▶"):
                st.write(f"{msg}  ·  {time.strftime('%H:%M:%S')}")

        def sleeper(seconds: float):
            end = time.time() + seconds
            while (left := end - time.time()) > 0:
                wait_box.warning(f"⏳ Waiting {left:.0f}s for free-tier rate limits — not frozen.")
                time.sleep(min(1.0, left))
            wait_box.empty()

        pipe = Pipeline(cfg, progress=progress, on_wait=lambda s, r: wait_box.warning(f"⏳ {r}: waiting {s:.0f}s"),
                        sleeper=sleeper)
        try:
            res = pipe.resume(resume_id) if resume_id else pipe.run(src, options)
            status.update(label=f"Done in {res['stats']['wall_clock_s']} s", state="complete", expanded=False)
            return res
        except QuotaPaused as e:
            status.update(label="Paused: free-tier quota exhausted", state="error")
            st.warning(str(e))
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
    return None


if ss.get("resume_id"):
    rid = ss.pop("resume_id")
    res = run_pipeline(None, rid)
    if res:
        ss.result, ss.seek = res, 0

if st.button("Summarize", type="primary", disabled=not source):
    res = run_pipeline(source)
    if res:
        ss.result, ss.seek, ss.quiz_checked = res, 0, False

# -------------------------------------------------------------------- result
res = ss.result
if not res:
    st.stop()

final, src, run_dir = res["final"], res["source"], cfg.path("runs_dir") / res["run_id"]
st.divider()
st.subheader(final.get("title") or src.get("title"))
if final.get("extractive_only"):
    st.warning("No LLM was available — showing an **extractive** (TextRank) summary and notes skeleton.")

col_v, col_d = st.columns([3, 1])
with col_v:
    if src.get("kind") == "youtube":
        st.video(src["source"], start_time=int(ss.seek))
    elif src.get("video_path") and Path(src["video_path"]).exists():
        st.video(src["video_path"], start_time=int(ss.seek))
with col_d:
    st.markdown("**Download**")
    for kind, mime, fname in (("markdown", "text/markdown", "lecture.md"), ("pdf", "application/pdf", "lecture.pdf"),
                              ("json", "application/json", "lecture.json")):
        p = run_dir / fname
        if p.exists():
            st.download_button(kind.upper(), p.read_bytes(), file_name=f"{res['run_id']}_{fname}", mime=mime,
                               width="stretch")
    v = res.get("verification") or {}
    if v.get("faithfulness") is not None:
        st.metric("Faithfulness", f"{v['faithfulness'] * 100:.0f}%", help="Key points supported by transcript/slide evidence (local NLI)")
    st.metric("LLM calls", res["stats"]["llm_calls"], help=f"cache hits: {res['stats']['cache_hits']}")


def seek_button(t, key, label=None):
    if t is None:
        return
    if st.button(label or f"▶ {fmt_ts(t)}", key=key):
        ss.seek = int(t)
        st.rerun()


tabs = st.tabs(["Summary", "Key Points", "Study Notes", "Slides", "Quiz", "Chapters", "Transcript", "Analytics"])
with tabs[0]:
    st.markdown(final.get("concise_summary", ""))
    if final.get("detailed_summary"):
        with st.expander("Detailed summary"):
            st.markdown(final["detailed_summary"])
    if final.get("glossary"):
        with st.expander(f"Glossary ({len(final['glossary'])} terms)"):
            for g in final["glossary"]:
                st.markdown(f"- **{g['term']}** — {g.get('definition', '')}")

with tabs[1]:
    for i, kp in enumerate(final.get("key_points", [])):
        c1, c2 = st.columns([10, 1.4])
        badge = BADGE.get(kp.get("status") or "", "")
        ev = {"slide": "🖼️ slide", "speech": "🗣️ speech", "both": "🖼️+🗣️"}.get(kp.get("evidence"), "")
        c1.markdown(f"{badge} {kp['text']}  \n<span style='opacity:.6;font-size:.85em'>{ev}</span>", unsafe_allow_html=True)
        with c2:
            seek_button(kp.get("start"), f"kp_{i}")
        if kp.get("evidence_text"):
            with st.expander("Evidence"):
                st.caption(f"{kp.get('status')} · NLI entailment {kp.get('support_score')} · "
                           f"{kp.get('evidence_source')} at {fmt_ts(kp.get('evidence_time'))}")
                st.write(kp["evidence_text"])
                if kp.get("status") == "unsupported":
                    st.warning("Not supported by the transcript or slides — check before relying on it.")

with tabs[2]:
    notes = final.get("study_notes_markdown", "")
    matches = match_slides_to_headings(notes, res.get("slides", []))
    buf: list[str] = []
    for i, line in enumerate(notes.splitlines()):
        buf.append(line)
        if i in matches:
            st.markdown("\n".join(buf))
            buf = []
            thumb = run_dir / matches[i]["thumbnail"]
            if thumb.exists():
                st.image(str(thumb), width=320, caption=f"Slide at {fmt_ts(matches[i]['start'])}")
    st.markdown("\n".join(buf))

with tabs[3]:
    slides = res.get("slides", [])
    if not slides:
        st.info(f"No slides detected (visual mode: {res.get('visual_mode')}) — speech-only topic segmentation was used.")
    cols = st.columns(3)
    for i, s in enumerate(slides):
        with cols[i % 3]:
            thumb = run_dir / s["thumbnail"]
            if thumb.exists():
                st.image(str(thumb), width="stretch")
            st.caption(f"**{s.get('title') or 'Slide'}** · {fmt_ts(s['start'])}–{fmt_ts(s['end'])} · OCR conf {s['ocr_confidence']:.2f}")
            with st.expander("OCR text"):
                st.text(s["text"] or "(no text)")
            seek_button(s["start"], f"sl_{i}")

with tabs[4]:
    rq = final.get("revision_questions") or {}
    for i, q in enumerate(rq.get("mcq", [])):
        st.markdown(f"**Q{i + 1}. {q['question']}**")
        letters = "ABCDEFGH"
        choice = st.radio("answer", [f"{letters[j]}) {o}" for j, o in enumerate(q["options"])], key=f"mcq_{i}",
                          index=None, label_visibility="collapsed")
        if ss.quiz_checked:
            ok = choice is not None and choice[0] == q["answer"]
            (st.success if ok else st.error)(f"{'Correct' if ok else 'Answer: ' + q['answer']} — {q.get('explanation', '')}")
    if rq.get("mcq") and st.button("Check answers"):
        ss.quiz_checked = True
        st.rerun()
    for i, q in enumerate(rq.get("short_answer", [])):
        st.markdown(f"**S{i + 1}. {q['question']}**")
        with st.expander("Model answer"):
            st.write(q["answer"])
    if not rq.get("mcq") and not rq.get("short_answer"):
        st.info("No quiz (extractive mode).")

with tabs[5]:
    for i, ch in enumerate(final.get("chapters", [])):
        c1, c2 = st.columns([10, 1.4])
        c1.markdown(f"**{ch['title']}**  \n<span style='opacity:.6'>{fmt_ts(ch.get('start'))} – {fmt_ts(ch.get('end'))}</span>",
                    unsafe_allow_html=True)
        with c2:
            seek_button(ch.get("start"), f"ch_{i}")

with tabs[6]:
    tr = read_json(run_dir / "transcript.json") or {}
    st.caption(f"Source: {tr.get('source')} · real-time factor {tr.get('real_time_factor')}")
    query = st.text_input("Search transcript")
    segs = [s for s in tr.get("segments", []) if not query or query.lower() in s["text"].lower()]
    st.caption(f"{len(segs)} segments")
    st.dataframe(pd.DataFrame([{"time": fmt_ts(s["start"]), "text": s["text"]} for s in segs[:2000]]),
                 hide_index=True, width="stretch", height=420)

with tabs[7]:
    s = res["stats"]
    v = res.get("verification") or {}
    c = st.columns(5)
    c[0].metric("Compression", f"{s.get('compression_ratio') or '–'}×")
    c[1].metric("Faithfulness", f"{v['faithfulness'] * 100:.0f}%" if v.get("faithfulness") is not None else "–")
    c[2].metric("LLM calls", s["llm_calls"])
    c[3].metric("Tokens in / out", f"{s['input_tokens']:,} / {s['output_tokens']:,}")
    c[4].metric("Wall clock", f"{s['wall_clock_s']:.0f} s")
    st.caption(f"Providers: {', '.join(s['providers']) or 'none'} · retries {s['retries']} · fallbacks {s['fallbacks']} · "
               f"rate-limit waits {s['rate_limit_wait_s']} s · transcript: {res.get('transcript_source')} · "
               f"segmentation: {res.get('segmentation_mode')}")
    st.markdown("**Key-point evidence: slide vs speech**")
    st.bar_chart(pd.DataFrame({"key points": s.get("evidence_split", {})}), horizontal=True, height=180)
    log_ = read_json(run_dir / "run_log.json") or {}
    stg = [{"stage": k, "seconds": (rec or {}).get("seconds") or 0, "status": (rec or {}).get("status")}
           for k, rec in log_.get("stages", {}).items()]
    if stg:
        st.markdown("**Stage timings (first run)**")
        st.bar_chart(pd.DataFrame(stg).set_index("stage")["seconds"], horizontal=True, height=300)
        st.dataframe(pd.DataFrame(stg), hide_index=True, width="stretch")
