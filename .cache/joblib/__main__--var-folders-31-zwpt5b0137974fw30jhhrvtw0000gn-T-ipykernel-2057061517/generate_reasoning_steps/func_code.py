# first line: 1
@memory.cache
def generate_reasoning_steps(examples):
    prompts = [generate_prompt(example) for example in examples]
    reasoning_steps = []
    answers = []
    for prompt in prompts:
        try:
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
            response = pipe(prompt, max_new_tokens=256, do_sample=True, batch_size=4)
            reasoning, answer = extract_reasoning(response)
            reasoning_steps.append(reasoning)
            answers.append(answer)
        except Exception as e:
            print(f"Error during llama_pipeline call: {e}")
            reasoning_steps.append(["Error: Could not generate reasoning."])
            answers.append("Error: Could not generate answer.")
    return {"reasoning": reasoning_steps, "answer": answers}
