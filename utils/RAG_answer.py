# RAG_answer.py

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
    predicted_answer_multiple_choice = []

    pattern = re.compile(
        r'''
        (?:                                       # non-capturing group for the two prompts
          correct\ answers?\ is                   # “correct answer is” or “correct answers is”
          |please\ state\ only\ the\ letter        # or “Please state only the letter”
        )
        \s*[:]*\s*                                # any spaces or colons after the prompt
        (?:\r?\n\s*)*                             # skip any number of blank/indented lines
        ([A-E])                                   # capture exactly one letter A–E
        ''',
        flags=re.IGNORECASE | re.VERBOSE
    )

    for sample in predictions:
        gen = sample[0].get('generated_answer') or ""
        m = pattern.search(gen)
        if m:
            predicted_answer_multiple_choice.append(m.group(1))
        else:
            predicted_answer_multiple_choice.append("na")
    return predicted_answer_multiple_choice





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
    predicted_answer_true_false = []
    for sample in predictions:
        match = re.search(r'Please state only False or True:\s*\n*(True|False)\.*',
                          sample[0].get('generated_answer'), re.IGNORECASE)
        if match:
            answer = match.group(1)
            predicted_answer_true_false.append({"predicted_answer": answer})
        else:
            predicted_answer_true_false.append({"predicted_answer": "na"})
    return predicted_answer_true_false



def extract_multi_hop_answers(predictions):
    """
    Extracts multi-hop answers from generated responses and detects blocked outputs.

    Parameters:
        predictions (list): List of dicts containing 'generated_answer' fields.

    Returns:
        Tuple:
            - List of dicts with extracted 'predicted_answer'
            - Integer count of blocked responses
    """
    predicted_answers = []
    blocked_count = 0

    pattern = re.compile(
        r'### You are a medical expert and equipped to answer this specific question. Please answer the question and elaborate what steps you took:\s*(.+)',
        flags=re.DOTALL
    )

    for sample in predictions:
        gen = sample.get('generated_answer', "")
        match = pattern.search(gen)
        if match:
            answer = match.group(1).strip()
            if answer.lower().startswith("i'm sorry, but"):
                answer = "N/A"
                blocked_count += 1
            predicted_answers.append({"predicted_answer": answer})
        else:
            predicted_answers.append({"predicted_answer": "na"})

    return predicted_answers, blocked_count

