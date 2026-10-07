from datasets import load_dataset

datasets=[
    "pile-of-law/pile-of-law",
    "sentence-transformers/codesearchnet",
    "AlgorithmicResearchGroup/s2orc_arxiv"
]

for name in datasets:
    print("\n==============================")
    print(name)
    print("==============================")

    dataset=load_dataset(
        name,
        split="train",
        streaming=True
    )

    example=next(iter(dataset))
    print(example)

