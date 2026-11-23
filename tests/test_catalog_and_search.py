import pytest

from catalog import exam_type, parse_header
from hybrid import BM25Index, reciprocal_rank_fusion


@pytest.mark.parametrize("header, expected", [
    ("Amrita Vishwa Vidyapeetham\nSixth Semester\n19CSE435 Computer Vision\nTime: Three hours",
     ("19CSE435", "Computer Vision")),
    ("B.Tech First Assessment\n15ENG111Communicative English\n[Time: Two hours",
     ("15ENG111", "Communicative English")),
    ("Third Semester\nCH329 / CHY273 Polymer Chemistry (Common to all branches)",
     ("CH329 / CHY273", "Polymer Chemistry")),
    ("ECE481- Agent Based Modeling-End Semester Examination\nCommon to 8th Semester",
     ("ECE481", "Agent Based Modeling")),
    ("CVL 201 SURVEYING\nTime: 3 hours", ("CVL201", "Surveying")),
    ("CO01 Understand the basics\nPage 1 of 2", (None, None)),
])
def test_parse_header(header, expected):
    assert parse_header(header) == expected


@pytest.mark.parametrize("text, expected", [
    ("Sixth Semester- Mid-Term April 2024", "Mid-Term"),
    ("Fifth Semester - Second Assessment Dec 2022 19CSE451 _II Ass Dec 2022.pdf", "Second Assessment"),
    ("15EEE211_II_Ass_March_2018.pdf", "Second Assessment"),
    ("AES222_AssI_Feb2016.pdf", "First Assessment"),
    ("Sixth Semester - July 2023 Regular 19CSE435_Supplementary.pdf", "Supplementary"),
    ("Fifth semester 2013 Regular CSE301.pdf", "End Semester"),
])
def test_exam_type(text, expected):
    assert exam_type(text) == expected


def test_bm25_prefers_exact_rare_terms():
    index = BM25Index([
        (1, "State and explain the Pilling-Bedworth rule"),
        (2, "Explain the electrochemical theory of corrosion"),
        (3, "Define the rule of mixtures for composites"),
    ])
    assert [doc_id for doc_id, _ in index.search("Pilling Bedworth rule")][0] == 1
    assert index.search("quantum tunnelling") == []


def test_reciprocal_rank_fusion_rewards_agreement():
    # "y" is second in one list and first in the other; "x" and "z" appear once each
    assert reciprocal_rank_fusion(["x", "y"], ["y", "z"]) == ["y", "x", "z"]
