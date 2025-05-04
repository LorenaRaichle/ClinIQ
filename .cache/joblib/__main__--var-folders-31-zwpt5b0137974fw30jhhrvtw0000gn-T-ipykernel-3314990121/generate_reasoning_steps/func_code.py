# first line: 1
@memory.cache
def generate_reasoning_steps(examples):
    prompts = [
        generate_prompt({"question": q, "answer": a})
        for q, a in zip(examples["question"], examples["answer"])
    ]
    
    # MOCK processing (replace with real logic if needed)
    reasoning_steps = ["Reasoning steps for: " + p for p in prompts]
    answers = ["Final answer for: " + p for p in prompts]

    return {
        "prompt": prompts,
        "reasoning": reasoning_steps,
        "answer": answers
    }
