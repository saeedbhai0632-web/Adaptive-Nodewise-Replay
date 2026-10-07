import torch
from transformers import AutoTokenizer
from torch.optim import AdamW

MODEL_PATH="./pretrained/Qwen2.5-1.5B"

def train(
    model,
    mixed_data,
    batch_size=2,
    max_length=512,
    steps=10,
    lr=1e-5
):
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Device:",device)

    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    model.to(device)
    model.train()

    optimizer=AdamW(model.parameters(),lr=lr)

    required_examples=steps*batch_size

    if len(mixed_data)<required_examples:
        raise ValueError(
            f"Need {required_examples} examples but only "
            f"{len(mixed_data)} were provided"
        )

    print("Training examples:",len(mixed_data))
    print("Examples consumed:",required_examples)

    for step in range(steps):
        start=step*batch_size
        batch=mixed_data[start:start+batch_size]

        batch=[x[1] for x in batch]

        inputs=tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids=inputs["input_ids"].to(device)
        attention_mask=inputs["attention_mask"].to(device)

        labels=input_ids.clone()
        labels[attention_mask==0]=-100

        optimizer.zero_grad()

        outputs=model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss=outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}/{steps} | Loss: {loss.item():.4f}")

    return model