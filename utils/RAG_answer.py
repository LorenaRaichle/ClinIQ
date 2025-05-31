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



def extract_multi_hop_answers(predictions):
    """
    Extracts free-text answers for multi-hop questions from model-generated responses.

    Looks for a specific prompt prefix and handles blocked or missing answers.

    Parameters:
        predictions (list): List of dicts containing 'generated_answer'.

    Returns:
        Tuple[list, int]: (list of extracted answers, count of blocked responses)
    """
    pattern = re.compile(
        r'### You are a medical expert and equipped to answer this specific question\. Please answer the question and elaborate what steps you took:\s*(.+)',
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
