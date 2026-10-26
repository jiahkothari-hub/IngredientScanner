"""Tests for the text-representation additions: ingredient2vec tokens and embeddings, the similarity
measures compared in src/evaluation/product_similarity.py, and the app's similar-product estimate.
Run:  pytest tests/test_text_representations.py -q"""
import numpy as np
import pytest
from scipy import sparse

from src.embeddings.ingredient2vec import MODEL_PATH, document_vector, record_tokens, term_token
from src.evaluation.product_similarity import cosine, jaccard

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run python -m src.embeddings.ingredient2vec first")


# ------------------------------------------------------------------ tokens
def test_term_token_normalises_like_the_dictionaries():
    assert term_token("Glucose  Syrup") == "glucose_syrup"
    assert term_token("Mono - and Diglycerides") == "mono-and_diglycerides"


def test_record_tokens_keep_reading_order_and_skip_codes():
    record = {"entities": [{"start": 20, "text": "Palm Oil", "label": "FAT"},
                           {"start": 0, "text": "Sugar", "label": "SUGAR"},
                           {"start": 10, "text": "INS 330", "label": "INS_CODE"}]}
    assert record_tokens(record) == ["sugar", "palm_oil"]


def test_document_vector_is_mean_of_known_tokens_and_zero_otherwise():
    wv = {"a": np.array([1.0, 0.0], dtype=np.float32), "b": np.array([0.0, 1.0], dtype=np.float32)}

    class Vectors(dict):
        vector_size = 2
    vectors = Vectors(wv)
    assert np.allclose(document_vector(["a", "b", "unknown"], vectors), [0.5, 0.5])
    assert not document_vector(["unknown"], vectors).any()


# ------------------------------------------------------------------ similarity measures
def test_jaccard_on_word_sets():
    q = sparse.csr_matrix(np.array([[1, 1, 0, 0]]))           # {w0, w1}
    d = sparse.csr_matrix(np.array([[2, 1, 0, 0],             # same set, different counts -> 1.0
                                    [1, 0, 1, 0],             # {w0, w2}: 1 shared of 3 -> 1/3
                                    [0, 0, 1, 1]]))           # nothing shared -> 0
    assert np.allclose(jaccard(q, d), [[1.0, 1 / 3, 0.0]])


def test_cosine_ignores_length():
    q = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0]]))
    d = sparse.csr_matrix(np.array([[3.0, 3.0, 0.0], [0.0, 0.0, 1.0]]))
    assert np.allclose(cosine(q, d), [[1.0, 0.0]])


# ------------------------------------------------------------------ trained model and app
@needs_model
def test_embeddings_put_sweeteners_together():
    from src.embeddings.ingredient2vec import most_similar
    neighbours = [t for t, _ in most_similar("sucralose", topn=5)]
    assert "aspartame" in neighbours


@needs_model
def test_estimate_uses_embeddings_and_finds_the_product():
    from src.app.nutrition import estimate_from_similar_product
    text = ("Sugar, Palm Oil, Hazelnuts (13%), Skimmed Milk Powder (8.7%), Fat-Reduced Cocoa (7.4%), "
            "Emulsifier: Lecithins (Soya), Vanillin.")
    estimate = estimate_from_similar_product(text)
    assert estimate["method"] == "TF-IDF + ingredient2vec"
    assert "nutella" in estimate["product_name"].lower()
    assert estimate["per_100g"]["sugars_g"] > 40


def test_estimate_refuses_unrelated_text():
    from src.app.nutrition import estimate_from_similar_product
    assert estimate_from_similar_product("qwerty zxcvb plmok") == {}
