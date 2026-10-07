from datasets import load_dataset

def get_dataset(node,max_samples=1000,start_index=0):
    if node==1:
        dataset=load_dataset(
            "slinusc/PubMedAbstractsSubset",
            split="train",
            streaming=True
        )

    elif node==2:
        dataset=load_dataset(
            "overthelex/indian-court-decisions",
            "high_courts",
            split="train",
            streaming=True
        )

    elif node==3:
        dataset=load_dataset(
            "code-search-net/code_search_net",
            split="train",
            streaming=True
        )

    elif node==4:
        dataset=load_dataset(
            "AlgorithmicResearchGroup/s2orc_arxiv",
            split="train",
            streaming=True
        )

    else:
        raise ValueError("Invalid node")

    data=[]

    for i,example in enumerate(dataset):
        if(i<start_index):
            continue
        if node==1:
            text=example["title"]+"\n"+example["abstract"]

        elif node==2:
            text=example["full_text"]

        elif node==3:
            text=example["func_code_string"]

        elif node==4:
            text=example["text"]

        if text is not None and len(text)>0:
            data.append(text)

        if len(data)>=max_samples:
            break

    return data