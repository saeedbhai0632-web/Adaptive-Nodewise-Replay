import json
import os
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import get_dataset
from src.replay import get_replay_dataset
from src.mixer import get_mixed_dataset
from src.trainer import train
from src.evaluator import evaluate_model
from src.federated import fedavg

MODEL_PATH="./pretrained/Qwen2.5-1.5B"
RHO_GRID=[0.0,0.05,0.10,0.20,0.40]
MAX_LENGTH=512


def nodewise_controller(rho,forgetting,improvement,tau_f,tau_low,tau_s):
    """
    One grid step per round, following the project guide:

      forgetting > tau_f                          -> rho up
      forgetting < tau_low AND improvement < tau_s -> rho down
      otherwise (safety band)                      -> hold

    forgetting  = anchor loss of the node's local model - anchor loss of theta_0
    improvement = domain-val loss of the global model at round start
                  - domain-val loss of the node's local model after training
    """
    idx=RHO_GRID.index(rho)

    if forgetting>tau_f:
        idx=min(idx+1,len(RHO_GRID)-1)
    elif forgetting<tau_low and improvement<tau_s:
        idx=max(idx-1,0)

    return RHO_GRID[idx]


def _count_replay(mixed_data,tokenizer):
    texts=[t for tag,t in mixed_data if tag=="replay"]
    total_tokens=0

    for i in range(0,len(texts),32):
        enc=tokenizer(
            texts[i:i+32],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True
        )
        total_tokens+=sum(len(ids) for ids in enc["input_ids"])

    return len(texts),total_tokens


def run_nodereplay(
    nodes,
    rounds,
    local_steps,
    batch_size,
    init_rho,
    max_local_samples,
    max_replay_samples,
    anchor_data,
    domain_val,
    controller_cfg=None,
    adaptive=True,
    seed=42,
    history_path=None
):
    """
    Federated training where every node has its own replay ratio.

    adaptive=False keeps every node at init_rho (calibration mode):
    the per-node signals are still measured and logged.
    """
    # init_rho may be one float (same for all nodes) or a {node: rho} dict
    # (used for the per-node-oracle fixed baseline).
    if isinstance(init_rho,dict):
        init_rhos={n:init_rho[n] for n in nodes}
    else:
        init_rhos={n:init_rho for n in nodes}

    for n,r in init_rhos.items():
        if r not in RHO_GRID:
            raise ValueError(f"init_rho for node {n} must be in {RHO_GRID}, got {r}")

    if adaptive and controller_cfg is None:
        raise ValueError("controller_cfg required when adaptive=True")

    device="cuda" if torch.cuda.is_available() else "cpu"

    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    global_model=AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    global_model.to(device)

    def loss_of(model,data):
        return evaluate_model(
            model,tokenizer,device,data,MAX_LENGTH,batch_size
        )["loss"]

    replay_data=get_replay_dataset(max_replay_samples)

    base_anchor_loss=loss_of(global_model,anchor_data)
    global_domain_loss={n:loss_of(global_model,domain_val[n]) for n in nodes}

    print(f"\nBase anchor loss (theta_0): {base_anchor_loss:.4f}")

    rho=dict(init_rhos)
    history=[]

    # Local data is identical every round (static slice) -> stream it once.
    local_cache={n:get_dataset(n,max_local_samples) for n in nodes}

    for round_num in range(rounds):
        print(f"\n===== ROUND {round_num+1}/{rounds} =====")

        node_models=[]
        node_weights=[]
        node_records=[]

        for node in nodes:
            print(f"\n--- Node {node} | rho={rho[node]:.2f} ---")

            local_data=local_cache[node]

            mixed_data=get_mixed_dataset(
                local_data,
                replay_data,
                rho[node],
                total_examples=local_steps*batch_size,
                seed=seed+round_num*1000+node
            )

            replay_examples,replay_tokens=_count_replay(mixed_data,tokenizer)

            node_model=deepcopy(global_model)
            node_model=train(
                node_model,
                mixed_data,
                batch_size=batch_size,
                steps=local_steps
            )

            # train() leaves the last step's grads allocated; free them
            # (~1 model-size of GPU memory per node model held until FedAvg).
            node_model.zero_grad(set_to_none=True)

            local_anchor_loss=loss_of(node_model,anchor_data)
            local_domain_loss=loss_of(node_model,domain_val[node])

            forgetting=local_anchor_loss-base_anchor_loss
            improvement=global_domain_loss[node]-local_domain_loss

            print(
                f"Node {node}: forgetting F={forgetting:+.5f} | "
                f"improvement P={improvement:+.5f} | "
                f"replay examples={replay_examples}"
            )

            node_models.append(node_model)
            node_weights.append(local_steps*batch_size)

            node_records.append({
                "node":node,
                "rho":rho[node],
                "replay_examples":replay_examples,
                "replay_tokens":replay_tokens,
                "local_anchor_loss":local_anchor_loss,
                "forgetting":forgetting,
                "local_domain_loss":local_domain_loss,
                "improvement":improvement
            })

        global_model.load_state_dict(fedavg(node_models,node_weights))

        del node_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        global_anchor_loss=loss_of(global_model,anchor_data)
        global_domain_loss={
            n:loss_of(global_model,domain_val[n]) for n in nodes
        }

        for rec in node_records:
            node=rec["node"]

            if adaptive:
                new_rho=nodewise_controller(
                    rho[node],
                    rec["forgetting"],
                    rec["improvement"],
                    **controller_cfg
                )
            else:
                new_rho=rho[node]

            rec["next_rho"]=new_rho
            rho[node]=new_rho

        history.append({
            "round":round_num+1,
            "global_anchor_loss":global_anchor_loss,
            "global_forgetting":global_anchor_loss-base_anchor_loss,
            "global_domain_loss":{str(n):v for n,v in global_domain_loss.items()},
            "nodes":node_records
        })

        print(
            f"\nRound {round_num+1} done | global forgetting="
            f"{global_anchor_loss-base_anchor_loss:+.5f} | next rho="
            f"{[rho[n] for n in nodes]}"
        )

        if history_path:
            os.makedirs(os.path.dirname(history_path) or ".",exist_ok=True)
            with open(history_path,"w",encoding="utf-8") as f:
                json.dump(history,f,indent=2)

    return global_model,history