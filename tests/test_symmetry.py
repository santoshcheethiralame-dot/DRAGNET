from types import SimpleNamespace

from scope.symmetry import case_symmetry, near_duplicate_pairs, similarity, symmetry_rates


def chunk(cid, text):
    return SimpleNamespace(chunk_id=cid, text=text)


def test_similarity_extremes():
    assert similarity("the same words here", "the same words here") == 1.0
    assert similarity("alpha beta gamma", "delta epsilon zeta") == 0.0
    assert similarity("", "anything") == 0.0


def test_near_duplicates_found_and_ordered():
    chunks = [
        chunk("a", "Paris is the capital of France and sits on the Seine"),
        chunk("b", "Paris is the capital of France and sits on the river Seine"),
        chunk("c", "The mitochondria is the powerhouse of the cell"),
    ]
    pairs = near_duplicate_pairs(chunks, threshold=0.6)
    assert [(p[0], p[1]) for p in pairs] == [("a", "b")]
    assert pairs[0][2] > 0.8


def test_case_symmetry_touching_members():
    chunks = [
        chunk("a", "the answer is nineteen eighty four exactly"),
        chunk("b", "the answer is nineteen eighty four precisely"),
        chunk("c", "something entirely different and unrelated"),
    ]
    touching = case_symmetry(chunks, members=frozenset({"a"}), threshold=0.6)
    assert touching["has_pair"] and touching["touches_members"]
    away = case_symmetry(chunks, members=frozenset({"c"}), threshold=0.6)
    assert away["has_pair"] and not away["touches_members"]
    assert away["closest"] > 0.6


def test_rates_pool_over_cases():
    with_pair = {"has_pair": True, "touches_members": True}
    without = {"has_pair": False, "touches_members": False}
    rates = symmetry_rates([with_pair, without, without, without])
    assert rates["n"] == 4
    assert rates["pair_rate"] == 0.25
    assert rates["member_rate"] == 0.25
    assert symmetry_rates([])["pair_rate"] is None
