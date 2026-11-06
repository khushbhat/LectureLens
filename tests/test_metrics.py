from evaluation.metrics import (compression_ratio, concept_coverage, mean_std, rouge, slide_concept_recall,
                                slide_only_terms, wer)


def test_rouge_identical_and_disjoint():
    assert rouge("the cat sat on the mat.", "the cat sat on the mat.")["rouge1"] == 1.0
    assert rouge("alpha beta gamma.", "delta epsilon zeta.")["rouge2"] == 0.0


def test_concept_coverage_exact_stemmed():
    # all concepts match exactly (stemmed) => embedding model is never loaded
    res = concept_coverage("Gradients and learning rates matter; SGD uses mini-batches.",
                           ["gradient", "learning rate", "mini-batch"], embed_model="unused")
    assert res["coverage"] == 1.0
    assert concept_coverage("x", [], "unused")["coverage"] is None


def test_slide_only_terms_and_recall():
    slides = [{"title": "Optimizers", "lines": [{"text": "Optimizers", "conf": 0.9},
                                                  {"text": "Adaptive methods: Adam, RMSProp", "conf": 0.9},
                                                  {"text": "cosine annealing schedule", "conf": 0.9},
                                                  {"text": "garbage %%$ xq", "conf": 0.2}]}]
    transcript = "We talk about optimizers and adaptive methods for training."
    terms = slide_only_terms(slides, transcript)
    assert "adam" in terms and "rmsprop" in terms and "cosine annealing schedule" in terms
    assert "adaptive methods" not in terms        # spoken
    assert not any("garbage" in t for t in terms)  # low-confidence OCR line ignored
    assert slide_concept_recall("We use Adam with a cosine annealing schedule.", terms) == 2 / 3
    assert slide_concept_recall("nothing", []) is None


def test_wer_compression_meanstd():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert abs(wer("the cat sat", "the bat sat") - 1 / 3) < 1e-9
    assert compression_ratio("a " * 100, "b " * 10) == 10.0
    m, s = mean_std([1.0, 3.0, None])
    assert m == 2.0 and s > 0
