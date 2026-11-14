from indonesian_hoax_checker.ner.demo import read_demo_cases


def test_demo_cases_file_has_all_categories():
    cases = read_demo_cases("data/demo_cases.txt")
    categories = {c.split(".")[0] for c, _ in cases}
    assert categories == {str(i) for i in range(1, 9)}
    assert "" in [t for _, t in cases]  # input kosong
    assert any("\n" in t for _, t in cases)  # beberapa paragraf
