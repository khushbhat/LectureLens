from lecturelens.audio import merge_chunk_segments, plan_chunks
from lecturelens.transcribe import clean_caption_text, parse_captions

ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500 align:start position:0%

welcome<00:00:00.500><c> to</c><00:00:01.000><c> the</c><00:00:01.500><c> course</c>

00:00:02.500 --> 00:00:02.510 align:start position:0%
welcome to the course


00:00:02.510 --> 00:00:05.000 align:start position:0%
welcome to the course
today<00:00:03.000><c> we</c><c> study</c><c> graphs</c>

00:00:05.000 --> 00:00:05.010 align:start position:0%
today we study graphs


00:00:05.010 --> 00:00:07.000 align:start position:0%
today we study graphs
[Music]

00:00:07.000 --> 00:00:09.000 align:start position:0%
&gt;&gt; a graph has &lt;b&gt;vertices&lt;/b&gt;
"""

SRT = """1
00:00:01,000 --> 00:00:03,000
<i>Hello</i> class.

2
00:00:03,500 --> 00:00:06,000
Let's begin [Applause]
"""


def test_rolling_captions_deduplicated():
    segs = parse_captions(ROLLING_VTT)
    texts = [s["text"] for s in segs]
    assert texts == ["welcome to the course", "today we study graphs", "a graph has vertices"]
    assert segs[0]["start"] == 0.0
    assert segs[1]["start"] == 2.51
    assert all(a["end"] <= b["start"] for a, b in zip(segs, segs[1:]))
    assert [s["id"] for s in segs] == [0, 1, 2]


def test_srt_and_noise_removed():
    segs = parse_captions(SRT)
    assert [s["text"] for s in segs] == ["Hello class.", "Let's begin"]
    assert segs[1]["start"] == 3.5 and segs[1]["end"] == 6.0


def test_clean_caption_text():
    assert clean_caption_text("<c>hi</c> &amp; [Music] ♪♪ bye") == "hi & bye"


def test_plan_chunks_overlap_and_coverage():
    chunks = plan_chunks(1500, 600, 3)
    assert chunks[0] == (0.0, 600)
    assert chunks[1][0] == 597          # 3 s overlap
    last_start, last_dur = chunks[-1]
    assert abs(last_start + last_dur - 1500) < 1e-6
    assert plan_chunks(300, 600, 3) == [(0.0, 300)]


def test_merge_chunks_offsets_and_overlap_dedup():
    c0 = [{"start": 0.0, "end": 5.0, "text": "First sentence here."},
          {"start": 595.0, "end": 599.5, "text": "and the gradient points uphill"}]
    # chunk 1 starts at 597 s: first segment repeats the overlap words
    c1 = [{"start": 0.0, "end": 1.2, "text": "uphill"},                       # fully inside covered audio
          {"start": 1.0, "end": 6.0, "text": "points uphill so we step down."},  # overlapping words
          {"start": 6.0, "end": 9.0, "text": "Next idea."}]
    merged = merge_chunk_segments([(0.0, c0), (597.0, c1)])
    texts = [m["text"] for m in merged]
    assert texts == ["First sentence here.", "and the gradient points uphill", "so we step down.", "Next idea."]
    assert merged[2]["start"] == 599.5          # clamped to end of covered audio
    assert merged[3]["start"] == 603.0          # offset applied
    assert [m["id"] for m in merged] == [0, 1, 2, 3]
