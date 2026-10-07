print("started")
from datasets import load_dataset

dataset=load_dataset(
    "slinusc/PubMedAbstractsSubset",
    split="train",
    streaming=True
)

for i,example in enumerate(dataset):
    print("Example",i)
    print(example)
    print()
    
    if i==2:
        break