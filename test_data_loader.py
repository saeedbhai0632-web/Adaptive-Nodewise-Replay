from src.data import get_loader

loader=get_loader(max_samples=10,batch_size=2)

for batch in loader:
    print("input_ids shape:",batch["input_ids"].shape)
    print("attention_mask shape:",batch["attention_mask"].shape)
    print("First tokenized example:")
    print(batch["input_ids"][0])
    break