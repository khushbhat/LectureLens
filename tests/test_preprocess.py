from lecturelens.preprocess import collapse_repeats, preprocess, remove_fillers


def test_fillers_removed_but_words_kept():
    out = remove_fillers("Um, so the uh umbrella, you know, works. Hmm.", ["um", "uh", "hmm", "you know"])
    assert "umbrella" in out and "Um" not in out and " uh " not in out and "you know" not in out


def test_collapse_repetition_loops():
    assert collapse_repeats("the the the cat") == "the cat"
    assert collapse_repeats("thank you thank you thank you for watching") == "thank you for watching"
    assert collapse_repeats("so so we begin") == "so we begin"


def test_sentence_timestamps(cfg, tmp_path):
    tr = {"segments": [
        {"id": 0, "start": 0.0, "end": 4.0, "text": "Um gradient descent minimizes loss. It uses"},
        {"id": 1, "start": 4.0, "end": 8.0, "text": "the gradient. the the The learning rate matters."},
    ]}
    res = preprocess(tmp_path, tr, cfg)
    sents = res["sentences"]
    assert sents[0]["text"].startswith("gradient descent minimizes loss")
    assert sents[0]["start"] == 0.0
    # "It uses the gradient." spans both segments
    span = next(s for s in sents if s["text"].startswith("It uses"))
    assert span["start"] < 4.0 < span["end"]
    assert all(s["start"] <= s["end"] for s in sents)
    assert (tmp_path / "sentences.json").exists()


def test_long_unpunctuated_run_split_with_short_tail(cfg, tmp_path):
    # 50 words, no punctuation: splits into 45 + 5 words, short tail merged back (regression)
    tr = {"segments": [{"id": 0, "start": 0.0, "end": 25.0, "text": " ".join(f"word{i}" for i in range(50))}]}
    sents = preprocess(tmp_path, tr, cfg)["sentences"]
    assert len(sents) == 1 and len(sents[0]["text"].split()) == 50
