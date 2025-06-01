# RAG_answer.py

import re


import re

def extract_multiple_choice_letters(predictions):
    """
    Extracts predicted answer letter (A–E) from model-generated answers.
    Returns 'na' if no valid option is found.
    """
    extracted = []

    for sample in predictions:
        answer = sample.get("generated_answer", "").strip()


        match = re.match(r"^\s*([A-E])[\.\s:\n]", answer, flags=re.IGNORECASE)
        if match:
            extracted.append(match.group(1).upper())
            continue


        match = re.search(r"\b([A-E])[\.\s:]\s", answer, flags=re.IGNORECASE)
        if match:
            extracted.append(match.group(1).upper())
            continue


        match = re.search(r"\b([A-E])\b", answer)
        if match:
            extracted.append(match.group(1).upper())
            continue

        extracted.append("na")

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
