import math
import os
import json
import csv
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

MODEL_PATH="./pretrained/Qwen2.5-1.5B"

def _get_model_and_tokenizer(model_path):
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model,tokenizer,device

def evaluate(model_path,data,max_length=512,batch_size=2):
    model,tokenizer,device=_get_model_and_tokenizer(model_path)
    return evaluate_model(model,tokenizer,device,data,max_length,batch_size)

def evaluate_model(model,tokenizer,device,data,max_length=512,batch_size=2):
    if len(data)==0:
        raise ValueError("Evaluation dataset is empty.")

    total_loss=0.0
    total_tokens=0

    model.eval()

    with torch.no_grad():
        for i in range(0,len(data),batch_size):
            batch=data[i:i+batch_size]

            inputs=tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True
            )

            input_ids=inputs["input_ids"].to(device)
            attention_mask=inputs["attention_mask"].to(device)

            labels=input_ids.clone()
            labels[attention_mask==0]=-100

            valid_tokens=(labels!=-100).sum().item()

            if valid_tokens==0:
                continue

            outputs=model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss+=outputs.loss.item()*valid_tokens
            total_tokens+=valid_tokens

    if total_tokens==0:
        raise ValueError("No valid tokens found in evaluation dataset.")

    avg_loss=total_loss/total_tokens

    return {
        "loss":avg_loss,
        "tokens":total_tokens,
        "perplexity":math.exp(min(avg_loss,20))
    }
def calculate_metrics(
    general_before,
    general_after,
    domain_before,
    domain_after,
    replay_ratio=None,
    replay_tokens=None,
    training_seconds=None
):
    forgetting=general_after-general_before
    domain_improvement=domain_before-domain_after

    result={
        "general_loss_before":general_before,
        "general_loss_after":general_after,
        "forgetting":forgetting,
        "domain_loss_before":domain_before,
        "domain_loss_after":domain_after,
        "domain_improvement":domain_improvement,
        "replay_ratio":replay_ratio,
        "replay_tokens":replay_tokens,
        "training_seconds":training_seconds
    }

    if general_before!=0:
        result["general_loss_change_pct"]=100*(general_after-general_before)/general_before
    else:
        result["general_loss_change_pct"]=None

    if domain_before!=0:
        result["domain_loss_change_pct"]=100*(domain_after-domain_before)/domain_before
    else:
        result["domain_loss_change_pct"]=None

    if replay_tokens is not None and replay_tokens>0:
        result["domain_improvement_per_replay_token"]=domain_improvement/replay_tokens
    else:
        result["domain_improvement_per_replay_token"]=None

    return result

def evaluate_condition(
    base_model_path,
    trained_model_path,
    general_data,
    domain_data,
    node,
    replay_ratio,
    replay_tokens=None,
    training_seconds=None,
    output_dir="./experiments/results",
    max_length=512,
    batch_size=2
):
    os.makedirs(output_dir,exist_ok=True)

    base_model,tokenizer,device=_get_model_and_tokenizer(base_model_path)
    general_before=evaluate_model(
        base_model,tokenizer,device,general_data,max_length,batch_size
    )
    domain_before=evaluate_model(
        base_model,tokenizer,device,domain_data,max_length,batch_size
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_model,tokenizer,device=_get_model_and_tokenizer(trained_model_path)
    general_after=evaluate_model(
        trained_model,tokenizer,device,general_data,max_length,batch_size
    )
    domain_after=evaluate_model(
        trained_model,tokenizer,device,domain_data,max_length,batch_size
    )
    del trained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics=calculate_metrics(
        general_before["loss"],
        general_after["loss"],
        domain_before["loss"],
        domain_after["loss"],
        replay_ratio,
        replay_tokens,
        training_seconds
    )

    metrics.update({
        "node":node,
        "general_ppl_before":general_before["perplexity"],
        "general_ppl_after":general_after["perplexity"],
        "domain_ppl_before":domain_before["perplexity"],
        "domain_ppl_after":domain_after["perplexity"],
        "general_eval_tokens":general_after["tokens"],
        "domain_eval_tokens":domain_after["tokens"]
    })

    safe_rho="none" if replay_ratio is None else f"{replay_ratio:.2f}".replace(".","")
    result_path=os.path.join(
        output_dir,
        f"node{node}_rho{safe_rho}.json"
    )

    with open(result_path,"w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2)

    print_results(metrics)
    return metrics

def print_results(metrics):
    print("\n===== EXPERIMENT RESULT =====")
    print(f"Node:                 {metrics['node']}")
    print(f"Replay ratio:         {metrics['replay_ratio']}")
    print(f"General loss:         {metrics['general_loss_before']:.4f} -> {metrics['general_loss_after']:.4f}")
    print(f"Forgetting:           {metrics['forgetting']:+.4f}")
    print(f"Domain loss:          {metrics['domain_loss_before']:.4f} -> {metrics['domain_loss_after']:.4f}")
    print(f"Domain improvement:   {metrics['domain_improvement']:+.4f}")
    print(f"Replay tokens:        {metrics['replay_tokens']}")
    if metrics["training_seconds"] is not None:
        print(f"Training time:        {metrics['training_seconds']:.2f}s")
    print(f"General PPL:           {metrics['general_ppl_before']:.2f} -> {metrics['general_ppl_after']:.2f}")
    print(f"Domain PPL:            {metrics['domain_ppl_before']:.2f} -> {metrics['domain_ppl_after']:.2f}")
    print("=============================\n")

def save_results(results,output_dir="./experiments/results"):
    os.makedirs(output_dir,exist_ok=True)
    path=os.path.join(output_dir,"all_results.csv")

    if not results:
        raise ValueError("No results supplied.")

    keys=[]
    for result in results:
        for key in result:
            if key not in keys:
                keys.append(key)

    with open(path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(output_dir,"all_results.json"),"w",encoding="utf-8") as f:
        json.dump(results,f,indent=2)

    print(f"Saved results to {path}")
    return path

def plot_experiment_a(results,output_dir="./experiments/results/plots"):
    os.makedirs(output_dir,exist_ok=True)

    nodes=sorted(set(r["node"] for r in results))
    rhos=sorted(set(r["replay_ratio"] for r in results))

    for node in nodes:
        rows=sorted(
            [r for r in results if r["node"]==node],
            key=lambda x:x["replay_ratio"]
        )

        x=[r["replay_ratio"] for r in rows]

        plt.figure()
        plt.plot(x,[r["forgetting"] for r in rows],marker="o")
        plt.axhline(0,linestyle="--")
        plt.xlabel("Replay ratio")
        plt.ylabel("General loss change")
        plt.title(f"Node {node}: forgetting vs replay ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,f"node{node}_forgetting.png"),dpi=200)
        plt.close()

        plt.figure()
        plt.plot(x,[r["domain_improvement"] for r in rows],marker="o")
        plt.axhline(0,linestyle="--")
        plt.xlabel("Replay ratio")
        plt.ylabel("Domain loss improvement")
        plt.title(f"Node {node}: domain improvement vs replay ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,f"node{node}_domain_improvement.png"),dpi=200)
        plt.close()

    plt.figure()
    for node in nodes:
        rows=sorted(
            [r for r in results if r["node"]==node],
            key=lambda x:x["replay_ratio"]
        )
        plt.plot(
            [r["forgetting"] for r in rows],
            [r["domain_improvement"] for r in rows],
            marker="o",
            label=f"Node {node}"
        )
    plt.axvline(0,linestyle="--")
    plt.axhline(0,linestyle="--")
    plt.xlabel("Forgetting (general loss change)")
    plt.ylabel("Domain loss improvement")
    plt.title("Experiment A: retention-specialization trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"retention_specialization_tradeoff.png"),dpi=200)
    plt.close()

    plt.figure()
    for node in nodes:
        rows=sorted(
            [r for r in results if r["node"]==node],
            key=lambda x:x["replay_ratio"]
        )
        plt.plot(
            [r["replay_ratio"] for r in rows],
            [r["replay_tokens"] if r["replay_tokens"] is not None else 0 for r in rows],
            marker="o",
            label=f"Node {node}"
        )
    plt.xlabel("Replay ratio")
    plt.ylabel("Replay tokens")
    plt.title("Experiment A: replay cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"replay_cost.png"),dpi=200)
    plt.close()

    print(f"Saved plots to {output_dir}")

def summarize_experiment_a(results):
    print("\n========== EXPERIMENT A SUMMARY ==========")

    for node in sorted(set(r["node"] for r in results)):
        rows=[r for r in results if r["node"]==node]

        best_retention=min(rows,key=lambda r:r["forgetting"])
        best_domain=max(rows,key=lambda r:r["domain_improvement"])

        # Simple balanced candidate: closest to the origin in the
        # (positive forgetting, positive domain-improvement) trade-off.
        balanced=min(
            rows,
            key=lambda r:max(r["forgetting"],0)**2+max(-r["domain_improvement"],0)**2
        )

        print(f"\nNode {node}:")
        print(f"  Best retention:  rho={best_retention['replay_ratio']}")
        print(f"    forgetting={best_retention['forgetting']:+.4f}")
        print(f"  Best domain improvement: rho={best_domain['replay_ratio']}")
        print(f"    improvement={best_domain['domain_improvement']:+.4f}")
        print(f"  Balanced candidate: rho={balanced['replay_ratio']}")

    print("\nInterpretation:")
    print("  Lower forgetting is better.")
    print("  Higher domain improvement is better.")
    print("  No automatic 'winner' is declared; inspect the trade-off plot.")
    print("===========================================\n")
