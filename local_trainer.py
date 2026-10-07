import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

print("STARTED")

device="cuda" if torch.cuda.is_available() else "cpu"
print("Using:",device)

model_path="./pretrained/Qwen2.5-0.5B"

tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

texts=[
    "The human body contains many different types of cells.",
    "Machine learning models learn patterns from data.",
    "The heart pumps blood throughout the body.",
    "Neural networks contain layers of connected neurons.",
    "Biomedical research helps us understand diseases."
]

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)

model.train()

for step in range(5):
    inputs=tokenizer(texts,return_tensors="pt",padding=True,truncation=True)
    inputs={k:v.to(device) for k,v in inputs.items()}

    labels=inputs["input_ids"].clone()
    labels[inputs["attention_mask"]==0]=-100

    outputs=model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels
    )

    loss=outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step+1}, Loss: {loss.item():.4f}")