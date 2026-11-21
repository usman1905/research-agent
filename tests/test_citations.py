from webresearch.citations import check, cited_ids, strip_invalid, support_score


def test_cited_ids_reads_groups():
    assert cited_ids("Claim [2, 1] and more [3].") == [1, 2, 3]


def test_support_score_ignores_markers():
    assert support_score("316L steel resists corrosion [1].", "316L stainless steel resists corrosion well") == 1.0
    assert support_score("Gold is cheap [1].", "steel is strong") == 0.0


def test_check_grades_each_sentence():
    sources = {1: "316L stainless steel resists corrosion and tarnish", 2: "Pakistan has a large population"}
    answer = "316L steel resists corrosion [1]. The moon is made of cheese [2]. Hello there. Bad ref [9]."
    statuses = [c.status for c in check(answer, sources)]
    assert statuses == ["supported", "weak", "uncited", "invalid"]


def test_strip_invalid_removes_unknown_numbers():
    assert strip_invalid("A [1, 7]. B [9]. C [2].", {1, 2}) == "A [1]. B. C [2]."
