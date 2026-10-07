from datasets import load_dataset

def get_replay_dataset(max_samples=1000,start_index=0):
    dataset=load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )

    data=[]

    for i,example in enumerate(dataset):
        if i<start_index:
            continue
        text=example["text"]

        if text is not None and len(text)>0:
            data.append(text)

        if len(data)>=max_samples:
            break

    return data