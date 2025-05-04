# first line: 1
@memory.cache
def generate_reasoning_steps(examples):
    prompts = [
        generate_prompt({"question": q, "answer": a})
        for q, a in zip(examples["question"], examples["answer"])
    ]
    reasoning_steps = []
    answers = []
    return {"prompt": prompts, "reasoning": reasoning_steps, "answer": answers}
