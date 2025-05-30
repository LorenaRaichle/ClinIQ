# utils/prompt_utils.py

def build_prompt(prompt_style: str, question: str):
    return prompt_style.replace("{_question_var_}", question)


def generate_model_input(example):
    if example['type'] == 'multiple_choice':
        prompt_style = """
        "{_question_var_}"

        ### What is the correct answer? Please state only the letter:
        """
        question = example['question']
        options = example['options']

        model_input = question + '\nOptions:\n'
        for key, val in options.items():
            model_input += f"{key}. {val}\n"

        example['prompt'] = build_prompt(prompt_style, model_input)
        example['answer'] = example.pop("correct_answer")
        example['prompt_n_answer'] = example['prompt'] + example['answer']

    elif example['type'] == 'true_false':
        prompt_style = """
        "{_question_var_}"

        ### Is this statement true or false? Please state only False or True:
        """
        example['prompt'] = build_prompt(prompt_style, example['question'])
        example['prompt_n_answer'] = example['prompt'] + example['answer']

    elif example['type'] == 'short_answer':
        prompt_style = """
        "{_question_var_}"

        ### You are a medical expert and equipped to answer this specific question. Please answer:
        """
        example['prompt'] = build_prompt(prompt_style, example['question'])
        example['prompt_n_answer'] = example['prompt'] + example['answer']

    elif example['type'] == 'multi_hop':
        prompt_style = """
        "{_question_var_}"

        ### You are a medical expert and equipped to answer this specific question. Please answer the question and elaborate what steps you took:
        """
        example['prompt'] = build_prompt(prompt_style, example['question'])
        example['prompt_n_answer'] = example['prompt'] + example['answer']

    return example
