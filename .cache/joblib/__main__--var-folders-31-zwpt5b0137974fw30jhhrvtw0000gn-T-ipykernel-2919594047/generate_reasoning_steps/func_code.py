# first line: 1
@memory.cache
def generate_reasoning_steps(examples):
    prompts = [
        format_few_shot_prompt(q, a)
        for q, a in zip(examples["question"], examples["answer"])
    ]
    reasoning_steps = []
    answers = []

    for prompt in prompts:
        try:
            response = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]["generated_text"]
            sentences = nltk.sent_tokenize(response)
            if not sentences:
                raise ValueError("No output")
            answer = sentences[-1]
            reasoning = [f"Step {i+1}: {sent.strip()}" for i, sent in enumerate(sentences[:-1])]
            reasoning_steps.append(reasoning)
            answers.append(answer)
        except Exception as e:
            print(f"Error during generation: {e}")
            reasoning_steps.append(["Error: Reasoning generation failed."])
            answers.append("Error: Answer generation failed.")
    
    return {"reasoning": reasoning_steps, "answer": answers}
