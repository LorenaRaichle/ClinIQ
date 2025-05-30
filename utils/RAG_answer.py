# RAG_answer.py

import re


def extract_multiple_choice_letters(predictions):
    """
    Extracts the predicted answer letter (A–E) from model-generated answers.

    Parameters:
        predictions (list): List of dicts containing 'generated_answer' fields.

    Returns:
        list: Extracted single-letter answers or 'na' if not found.
    """
    pattern = re.compile(
        r'''
        (?:correct\ answers?\ is|please\ state\ only\ the\ letter)
        \s*[:]*\s*
        (?:\r?\n\s*)*
        ([A-E])
        ''',
        flags=re.IGNORECASE | re.VERBOSE
    )

    extracted = []
    for sample in predictions:
        gen = sample.get('generated_answer') or ""
        m = pattern.search(gen)
        extracted.append(m.group(1) if m else "na")

    return extracted
