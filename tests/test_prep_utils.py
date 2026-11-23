from prep_utils import grounded_questions, split_parts, split_subparts


def q(text, marks=None):
    return {"question_text": text, "marks": marks}


def test_split_subparts_with_per_part_marks():
    parts = split_subparts(q("Compare turboprop and turbofan engines (6) b) Explain the terms orbital velocity and escape velocity. (4)", 10))
    assert parts == [q("Compare turboprop and turbofan engines", 6),
                     q("Explain the terms orbital velocity and escape velocity.", 4)]


def test_split_subparts_without_per_part_marks_sets_none():
    parts = split_subparts(q("Explain the mechanisms of laser micromachining. b) Discuss any five applications of laser micromachining.", 10))
    assert [p["marks"] for p in parts] == [None, None]
    assert len(parts) == 2


def test_mcq_options_are_not_split():
    mcq = q("Which of the following was NOT developed by B.F. Skinner?\n\na. The puzzle box\nb. Operant conditioning\nc. The cumulative recorder")
    assert split_subparts(mcq) == [mcq]
    short_options = q("Calculate the modes of vibration for the following molecules. a) SO b) HCl2 c) CH d) H O2", 2)
    assert split_subparts(short_options) == [short_options]


def test_grounded_questions_drops_questions_not_in_source_and_orders_by_position():
    source = "Part A\n1. Define entropy and state its unit.\n2. Explain the Carnot cycle with a neat diagram."
    questions = [
        q("Explain the Carnot cycle with a neat diagram.", 2),
        q("Derive the Clausius inequality for an irreversible process.", 10),   # not in source
        q("1. Define entropy and state its unit.", 2),
    ]
    assert [x["question_text"] for x in grounded_questions(source, questions)] == [
        "Define entropy and state its unit.",
        "Explain the Carnot cycle with a neat diagram.",
    ]


def test_split_parts():
    a, b = split_parts("header Part A (10 x 2) q1 q2 Part B (5 x 10) q3")
    assert a.startswith("Part A") and b.startswith("Part B")
    assert split_parts("no parts here") == ("no parts here", None)
