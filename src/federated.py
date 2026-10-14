from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM

from src.data import get_dataset
from src.replay import get_replay_dataset
from src.mixer import get_mixed_dataset
from src.trainer import train

MODEL_PATH="./pretrained/Qwen2.5-1.5B"


def fedavg(models,weights):
    if len(models)!=len(weights):
        raise ValueError("Number of models and weights must match")

    if len(models)==0:
        raise ValueError("No models provided")

    if any(w<0 for w in weights):
        raise ValueError("FedAvg weights must be non-negative")

    total=sum(weights)

    if total<=0:
        raise ValueError("Total FedAvg weight must be positive")

    weights=[w/total for w in weights]

    avg_state=deepcopy(models[0].state_dict())

    for key in avg_state:
        if avg_state[key].is_floating_point():
            avg_state[key]=sum(
                w*model.state_dict()[key]
                for model,w in zip(models,weights)
            )
        else:
            avg_state[key]=models[0].state_dict()[key]

    return avg_state


def run_federated(
    nodes,
    rounds=2,
    local_steps=2,
    batch_size=2,
    replay_ratio=0.10,
    max_local_samples=20,
    max_replay_samples=20,
    seed = 42
):
    device="cuda" if torch.cuda.is_available() else "cpu"

    global_model=AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    global_model.to(device)

    replay_data=get_replay_dataset(max_replay_samples)

    for round_num in range(rounds):
        print(f"\n===== ROUND {round_num+1}/{rounds} =====")

        node_models=[]
        node_weights=[]

        for node in nodes:
            print(f"\n--- Node {node} ---")

            local_data=get_dataset(
                node,
                max_local_samples
            )

            mixed_data=get_mixed_dataset(
                local_data,
                replay_data,
                replay_ratio,
                total_examples=local_steps*batch_size,
                seed=seed+round_num*1000+node
                
            )

            node_model=deepcopy(global_model)

            node_model=train(
                node_model,
                mixed_data,
                batch_size=batch_size,
                steps=local_steps
            )

            node_models.append(node_model)

            examples_used=local_steps*batch_size
            node_weights.append(examples_used)

            print(
                f"Node {node} training examples used: "
                f"{examples_used}"
            )

        print("\nFedAvg weights:")
        print(node_weights)

        global_state=fedavg(
            node_models,
            node_weights
        )

        global_model.load_state_dict(global_state)

        print(
            f"\nFedAvg completed for round "
            f"{round_num+1}"
        )

    return global_model


if __name__=="__main__":
    model=run_federated(
        nodes=[1,2,3,4],
        rounds=2,
        local_steps=2,
        replay_ratio=0.10,
        max_local_samples=20,
        max_replay_samples=20,
        seed=42
    )

    print("\nFederated training completed.")