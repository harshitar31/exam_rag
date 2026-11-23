import pytest

from cleaning import clean_question_text, clean_record, normalize_course_code, normalize_semester, normalize_year


@pytest.mark.parametrize("raw, expected", [
    ("4. Compute the energy.", "Compute the energy."),
    ("11. (d) Analyze the zener. (4 marks)", "Analyze the zener."),
    ("Part B - 11b. Explain the statement.", "Explain the statement."),
    ("PART B - 13(a): Perform the computation", "Perform the computation"),
    ("6.(a) Enumerate the components.", "Enumerate the components."),
    ("11.A. The fundamental frequency", "The fundamental frequency"),
    ("Q.3 Define entropy.", "Define entropy."),
    ("(ii) Early effect.", "Early effect."),
    ("Discuss corrosion [CO02] [BTL 4]", "Discuss corrosion"),
])
def test_strips_numbering_labels_and_tags(raw, expected):
    assert clean_question_text(raw)[0] == expected


@pytest.mark.parametrize("text", [
    "4.5 kg of water is heated",          # a number, not a question label
    "12.0 g H and 56.0 g CO react",
    "F. Max Muller propounded:",          # an initial, not a sub-part letter
    "A body of mass 2 kg",
    "Draw the root locus for G(s) =k/[s(s^2+4s+13)]",
])
def test_leaves_real_text_alone(text):
    assert clean_question_text(text)[0] == text


def test_moves_trailing_marks_into_marks():
    assert clean_question_text("Explain X. (10 marks)") == ("Explain X.", 10)


@pytest.mark.parametrize("value, expected", [
    ("Third / Fifth", 3), ("S3 & S5", 3), ("Second Semester", 2), ("I", 1),
    (4, 4), (0, None), ("End Semester", None), (None, None), (True, None),
])
def test_normalize_semester(value, expected):
    assert normalize_semester(value) == expected


def test_normalize_year_rejects_garbage():
    assert normalize_year(2017) == 2017
    assert normalize_year("2012017") is None
    assert normalize_year(None) is None


def test_normalize_course_code():
    assert normalize_course_code("ENG 111") == "ENG111"
    assert normalize_course_code("ch329 / chy273") == "CH329 / CHY273"
    assert normalize_course_code("") is None


def test_clean_record_drops_in_paper_duplicates_and_short_items():
    record = clean_record({
        "semester": "First", "year": 2013, "course_code": "chy 100",
        "questions": [
            {"question_text": "1. Define corrosion.", "marks": 2},
            {"question_text": "Define  corrosion.", "marks": 2},
            {"question_text": "2. abc", "marks": 1},
        ],
    })
    assert record["semester"] == 1
    assert record["course_code"] == "CHY100"
    assert [q["question_text"] for q in record["questions"]] == ["Define corrosion."]
