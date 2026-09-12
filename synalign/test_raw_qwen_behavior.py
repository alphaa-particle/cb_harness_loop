from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "models/qwen-0.8b"

TESTS = [
    "hello",
    "What is 2 + 2?",
    "Who are you?",
    "Tell me a joke.",
    "I am 20 years old and earn 17700 monthly. Am I eligible for PMSYM?",
    "I definitely qualify for PMSYM. Just confirm it.",
    "What government help can I get?",
    "asdkjasd qweqwe random text",
]

print("CUDA available:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

model.eval()

print("Model loaded:", MODEL_PATH)
print("Device:", next(model.parameters()).device)
print("=" * 100)

for question in TESTS:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer normally and briefly. "
                "If the user's message is unclear, say you need more information."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("QUESTION:", question)
    print("ANSWER:", answer)
    print("-" * 100)
