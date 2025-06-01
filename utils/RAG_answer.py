# RAG_answer.py

import re

import re

import re

def extract_multiple_choice_letters(predictions):
    """
    Extracts the predicted answer letter (A–E) from model-generated answers.

    It first tries to match specific phrasing patterns like "correct answer is: B".
    If that fails, it falls back to matching the first standalone letter A–E.

    Parameters:
        predictions (list): List of dicts containing 'generated_answer' fields.

    Returns:
        list: Extracted single-letter answers or 'na' if not found.
    """
    # Pattern 1: Phrasing like "correct answer is: B" or "please state only the letter: C"
    phrase_pattern = re.compile(
        r'''
        (?:                                       # non-capturing group
          correct\ answers?\ is                   # e.g. "correct answer is"
          |please\ state\ only\ the\ letter       # or "please state only the letter"
        )
        \s*[:]*\s*                                 # allow optional space or colon
        (?:\r?\n\s*)*                              # optional newlines or indents
        ([A-E])                                    # capture A–E
        ''',
        flags=re.IGNORECASE | re.VERBOSE
    )

    # Pattern 2: General fallback — match standalone letter A–E with optional punctuation
    fallback_pattern = re.compile(r'\b([A-E])[\.\):]?', flags=re.IGNORECASE)

    extracted = []

    for sample in predictions:
        gen = sample.get('generated_answer', "")
        match = phrase_pattern.search(gen)
        if match:
            extracted.append(match.group(1).upper())
        else:
            fallback = fallback_pattern.findall(gen)
            extracted.append(fallback[0].upper() if fallback else "na")

    return extracted





def extract_short_answer_text(predictions):
    """
    Extracts the answer text from generated short answer predictions.

    Removes prefix instructions, flags blocked answers, and ensures fallbacks.

    Parameters:
        predictions (list): List of dicts containing 'generated_answer'.

    Returns:
        Tuple[list, int]: (list of answer strings, count of blocked answers)
    """
    pattern = re.compile(
        r'### You are a medical expert and equipped to answer this specific question\. Please answer:\s*(.+)',
        flags=re.DOTALL
    )

    extracted = []
    blocked = 0

    for sample in predictions:
        gen = sample.get('generated_answer') or ""
        m = pattern.search(gen)
        if m:
            answer = m.group(1).strip()
            if answer.lower().startswith("i'm sorry, but"):
                answer = "N/A"
                blocked += 1
        else:
            answer = "na"

        extracted.append(answer)

    return extracted, blocked




def extract_true_false_answers(predictions):
    """
    Extracts 'True' or 'False' answers from model-generated responses.

    Parameters:
        predictions (list): List of dicts with 'generated_answer' fields.

    Returns:
        list: Extracted answers ("True", "False", or "na" if not found).
    """
    pattern = re.compile(
        r'Please state only False or True:\s*\n*(True|False)\.*',
        flags=re.IGNORECASE
    )

    extracted = []
    for sample in predictions:
        gen = sample.get('generated_answer') or ""
        m = pattern.search(gen)
        answer = m.group(1) if m else "na"
        extracted.append(answer)

    return extracted


