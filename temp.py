from joblib import Memory
from tqdm.auto import tqdm
import nltk

# Download necessary NLTK data if not already downloaded
nltk.download('punkt')

# Set pad_token_id to eos_token_id for open-end generation

# TODO: ? might be not needed
tokenizer.pad_token_id = tokenizer.eos_token_id

# Initialize caching
memory = Memory(location=".cache", verbose=0)

# Function to generate prompt
def generate_prompt(example):
    return f"""
    Question: {example['question']}
    Answer: {example['answer']}
    Provide a step-by-step reasoning breakdown explaining how the answer was derived.
    Each step should be clearly numbered and logically connected.
    """

# Function to extract reasoning and answer
def extract_reasoning(response):
    generated_text = response[0]["generated_text"]
    sentences = nltk.sent_tokenize(generated_text)
    answer = sentences[-1]
    reasoning = [f"Step {i+1}: {step.strip()}" for i, step in enumerate(sentences[:-1]) if step]
    return reasoning, answer

# Function to generate reasoning steps (with caching)
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

# Load the dataset
dataset = load_dataset("lavita/MedQuAD", split="train")

# Apply the function to the dataset
dataset = dataset.map(generate_reasoning_steps, batched=True, batch_size=4)

# Transform the dataset to the desired format
transformed_data_R1 = []
for item in tqdm(dataset, desc="Transforming data"):
    formatted_item = {
        "answer": item["answer"],
        "question": item["question"],
        "reasoning": item["reasoning"],
        "source": {
            "isbn": "000-0000000000",
            "page": 0,
            "paragraph_id": "000-0000000000-p00-para00"
        },
        "type": "multi_hop"
    }
    transformed_data_R1.append(formatted_item)

# Print the first 3 formatted entries
print(json.dumps(transformed_data_R1[:3], indent=4))
transformed_R1_data = transformed_data_R1