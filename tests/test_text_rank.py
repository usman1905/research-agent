from webresearch.rank import Passage, bm25, top_passages
from webresearch.text import passages, sentences, tokens


def test_tokens_drop_stopwords_and_case():
    assert tokens("The Quick brown FOX, and the dog") == ["quick", "brown", "fox", "dog"]


def test_sentences_keep_citations_attached():
    text = "Steel 316L resists corrosion [1]. It costs more [2]. Ok?"
    assert sentences(text) == ["Steel 316L resists corrosion [1].", "It costs more [2].", "Ok?"]


def test_passages_overlap_and_cover_the_text():
    words = [f"w{i}" for i in range(250)]
    chunks = passages(" ".join(words), size=100, overlap=20)
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w80"
    assert chunks[-1].split()[-1] == "w249"
    assert passages("") == []


def test_bm25_prefers_relevant_documents():
    docs = ["cats sleep all day", "stainless steel jewellery 316L", "steel bridges"]
    scores = bm25("316L stainless steel", docs)
    assert scores[1] > scores[2] > scores[0] == 0


def test_top_passages_limits_per_source_and_skips_irrelevant():
    cands = [Passage(0, "python agent search") for _ in range(5)] + [Passage(1, "agent search web"), Passage(2, "cooking")]
    best = top_passages("agent search", cands, k=10, per_source=2)
    assert [p.source_id for p in best].count(0) == 2
    assert all(p.source_id != 2 for p in best)
