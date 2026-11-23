"""
Helpers shared by the training-data builders (build_data_for_qs.py,
fix_training_data.py) and evaluate.py, so that training and evaluation see
PDF text in exactly the same format.
"""

import io
import re
import pdfplumber

from cleaning import clean_question_text

PAGE_BREAK = "\n\n--- PAGE BREAK ---\n\n"


def extract_pdf_text(pdf_bytes, first_page_max=1500, max_chars=8000):
    """Returns (first_page_text, full_text) with tables appended as 'a | b | c' rows."""
    pages = []
    first_page = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row if cell)
                    if row_text.strip():
                        text += "\n" + row_text
            text = text.strip()
            if i == 0:
                first_page = text[:first_page_max]
            pages.append(text)
    full_text = PAGE_BREAK.join(pages)
    return first_page, (full_text[:max_chars] if max_chars else full_text)


def split_parts(full_text):
    """Split paper into Part A and Part B. Returns (part_a, part_b or None)."""
    pattern = re.compile(r'(Part\s+[AB]\s*[\(\d])', re.IGNORECASE)
    splits = list(pattern.finditer(full_text))
    if len(splits) >= 2:
        return full_text[splits[0].start():splits[1].start()], full_text[splits[1].start():]
    elif len(splits) == 1:
        return full_text[splits[0].start():], None
    return full_text, None


# ---------------- grounding ----------------

def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _tokens(text):
    return re.findall(r"[a-z0-9]{3,}", text.lower())


def _ngrams(tokens, n=4):
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def grounded_questions(source_text, questions, min_token_coverage=0.8, min_ngram_overlap=0.3):
    """
    Returns the questions that actually appear in source_text, cleaned and in
    reading order. A training target must only contain questions the model can
    see in its input -- otherwise it learns to invent questions.

    A question counts as present if its first 30 (or 15) characters appear
    verbatim (ignoring spacing/punctuation), or -- to tolerate OCR noise and
    spelling fixes -- if most of its words AND some of its 4-word phrases do.
    """
    source_norm = _norm(source_text)
    source_token_list = _tokens(source_text)
    source_tokens = set(source_token_list)
    source_ngrams = _ngrams(source_token_list)

    found = []
    for order, q in enumerate(questions):
        text, text_marks = clean_question_text(q.get("question_text", ""))
        if len(text) < 10:
            continue
        n = _norm(text)
        position = -1
        for prefix_len in (30, 15):
            if len(n) >= prefix_len:
                position = source_norm.find(n[:prefix_len])
                if position >= 0:
                    break
        if position < 0:
            toks = _tokens(text)
            if not toks or len(set(toks) & source_tokens) / len(set(toks)) < min_token_coverage:
                continue
            grams = _ngrams(toks)
            if grams and len(grams & source_ngrams) / len(grams) < min_ngram_overlap:
                continue
            # Grounded by vocabulary but not verbatim (e.g. OCR spacing); place it
            # where its first long word appears.
            anchor = next((t for t in toks if len(t) >= 5), toks[0])
            position = source_norm.find(anchor)
        marks = q.get("marks") if isinstance(q.get("marks"), int) else text_marks
        found.append((position if position >= 0 else float("inf"), order,
                      {"question_text": text, "marks": marks}))

    found.sort(key=lambda item: (item[0], item[1]))
    # drop exact duplicates
    seen, out = set(), []
    for _, _, q in found:
        key = _norm(q["question_text"])
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


# ---------------- sub-parts ----------------

# "... b) Discuss", "... (c) Derive", or "b. Explain" at a line/sentence start
SUBPART_MARKER = re.compile(
    r"(?:(?<=\s)|^)(?:\(([b-h])\)|([b-h])\))\s+(?=[A-Z])"
    r"|(?:(?<=\n)|(?<=[.?:)]\s))([b-h])\.\s+(?=[A-Z])")
# an "a)" / "(a)" / "a." marker after the start means an MCQ option list or a
# stem with parts -- leave those alone
A_MARKER = re.compile(r"(?<=\s)(?:\(a\)|a\)|a\.)\s")
MIN_PART_CHARS = 25


def split_subparts(question):
    """
    The prompt asks for sub-parts (a), (b), (c) as separate questions, but ~7%
    of the gpt-oss labels merged them ("Explain X. b) Discuss Y. (6)"), which
    taught the model to merge too. Split when every part is a real question;
    otherwise return the question unchanged.
    """
    text = question["question_text"]
    if A_MARKER.search(text):
        return [question]
    markers = list(SUBPART_MARKER.finditer(text))
    letters = [next(g for g in m.groups() if g) for m in markers]
    if not markers or letters != [chr(ord("b") + i) for i in range(len(letters))]:
        return [question]

    bounds = [0] + [m.start() for m in markers] + [len(text)]
    starts = [0] + [m.end() for m in markers]
    parts = [clean_question_text(text[s:e]) for s, e in zip(starts, bounds[1:])]
    if any(len(t) < MIN_PART_CHARS for t, _ in parts):
        return [question]

    part_marks = [m for _, m in parts]
    if all(m is not None for m in part_marks):
        marks = part_marks
    else:
        # Only a combined mark for the whole question: per-part marks are unknown
        marks = [None] * len(parts)
    return [{"question_text": t, "marks": m} for (t, _), m in zip(parts, marks)]
