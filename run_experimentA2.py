import os
import time
import json
import random
import shutil
import argparse
import glob

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.data import get_dataset
from src.replay import get_replay_dataset
from src.federated import run_federated
from src.evaluator import evaluate_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH="./pretrained/Qwen2.5-1.5B"

NODES=[1,2,3,4]

RHO_GRID=[0.00,0.05,0.10,0.20,0.40]

SEEDS=[42,43,44]

ROUNDS=10

LOCAL_SAMPLES=1000
REPLAY_SAMPLES=1000
EVAL_SAMPLES=500

LOCAL_STEPS=100
BATCH_SIZE=2

MAX_LENGTH=512
LR=1e-5

TRAINING_EXAMPLES_PER_NODE_PER_ROUND=LOCAL_STEPS*BATCH_SIZE

RESULTS_DIR="./experiments/results/experimentA2"
CHECKPOINT_DIR="./experiments/checkpoints/experimentA2"

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SAVE_CHECKPOINTS=False   
MIN_FREE_GB=20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def check_disk_space(path,min_free_gb=MIN_FREE_GB):
    os.makedirs(path,exist_ok=True)
    free_gb=shutil.disk_usage(path).free/1e9
    if free_gb<min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free (< {min_free_gb} GB). Stopping cleanly."
        )
    return free_gb
def aggregate_results():
    files=sorted(glob.glob(
        os.path.join(RESULTS_DIR,"runs","*","result.json")
    ))
    results=[]
    for p in files:
        with open(p,"r",encoding="utf-8") as f:
            results.append(json.load(f))
    print(f"Found {len(results)}/{len(RHO_GRID)*len(SEEDS)} completed runs")
    save_all_results(results)
    summarize_results(results)
def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


def get_tokenizer():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    return tokenizer


def count_tokens(data,tokenizer,max_length=512):
    total_tokens=0

    for i in range(0,len(data),32):
        batch=data[i:i+32]

        inputs=tokenizer(
            batch,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )

        for ids in inputs["input_ids"]:
            total_tokens+=len(ids)

    return total_tokens


def save_checkpoint(model,checkpoint_path):
    os.makedirs(checkpoint_path,exist_ok=True)

    model.save_pretrained(checkpoint_path)

    tokenizer=get_tokenizer()
    tokenizer.save_pretrained(checkpoint_path)


def save_config():
    config={
        "experiment":"Experiment A2",
        "description":"Federated global fixed replay baseline",
        "model":MODEL_PATH,
        "nodes":NODES,
        "replay_ratios":RHO_GRID,
        "seeds":SEEDS,
        "rounds":ROUNDS,
        "local_samples":LOCAL_SAMPLES,
        "replay_samples":REPLAY_SAMPLES,
        "eval_samples":EVAL_SAMPLES,
        "local_steps":LOCAL_STEPS,
        "batch_size":BATCH_SIZE,
        "training_examples_per_node_per_round":
            TRAINING_EXAMPLES_PER_NODE_PER_ROUND,
        "max_length":MAX_LENGTH,
        "learning_rate":LR,
        "device":DEVICE
    }

    path=os.path.join(
        RESULTS_DIR,
        "experimentA2_config.json"
    )

    with open(path,"w",encoding="utf-8") as f:
        json.dump(config,f,indent=2)

    return path


def get_run_name(rho,seed):
    rho_name=f"{rho:.2f}".replace(".","")
    return f"global_rho{rho_name}_seed{seed}"


def run_already_completed(rho,seed):
    run_name=get_run_name(rho,seed)

    run_results_dir=os.path.join(
        RESULTS_DIR,
        "runs",
        run_name
    )

    if not os.path.exists(run_results_dir):
        return False

    result_file=os.path.join(
        run_results_dir,
        "result.json"
    )

    return os.path.exists(result_file)


# ============================================================
# EVALUATION HELPERS
# ============================================================

def evaluate_global_model(
    model,
    tokenizer,
    device,
    general_eval,
    domain_evals
):
    """
    Evaluate one final federated global model on:

    1. General held-out data
    2. Each node's held-out domain data
    """

    results={}

    print("\nEvaluating final global model on general data...")

    general_after=evaluate_model(
        model,
        tokenizer,
        device,
        general_eval,
        MAX_LENGTH,
        BATCH_SIZE
    )

    results["general_after"]=general_after

    print(
        f"General loss: "
        f"{general_after['loss']:.4f}"
    )

    results["domains"]={}

    for node in NODES:

        print(
            f"\nEvaluating final global model "
            f"on Node {node} domain..."
        )

        domain_after=evaluate_model(
            model,
            tokenizer,
            device,
            domain_evals[node],
            MAX_LENGTH,
            BATCH_SIZE
        )

        results["domains"][node]=domain_after

        print(
            f"Node {node} domain loss: "
            f"{domain_after['loss']:.4f}"
        )

    return results


# ============================================================
# BASELINE EVALUATION
# ============================================================

def evaluate_base_model(
    general_eval,
    domain_evals
):
    """
    Evaluate theta_0 once.

    The same theta_0 baseline is used for every A2 condition.
    """

    print("\n")
    print("============================================================")
    print("Evaluating base model theta_0")
    print("============================================================")

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_PATH
    )

    model.to(DEVICE)
    model.eval()

    tokenizer=get_tokenizer()

    results={}

    print("\nGeneral baseline...")

    general_before=evaluate_model(
        model,
        tokenizer,
        DEVICE,
        general_eval,
        MAX_LENGTH,
        BATCH_SIZE
    )

    results["general"]=general_before

    print(
        f"General loss: "
        f"{general_before['loss']:.4f}"
    )

    results["domains"]={}

    for node in NODES:

        print(
            f"\nNode {node} domain baseline..."
        )

        domain_before=evaluate_model(
            model,
            tokenizer,
            DEVICE,
            domain_evals[node],
            MAX_LENGTH,
            BATCH_SIZE
        )

        results["domains"][node]=domain_before

        print(
            f"Node {node} loss: "
            f"{domain_before['loss']:.4f}"
        )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================
# RESULT CONSTRUCTION
# ============================================================

def build_result(
    rho,
    seed,
    base_results,
    final_results,
    training_seconds,
    replay_pool_tokens,
    general_eval_tokens,
    domain_eval_tokens
):
    general_before=base_results["general"]
    general_after=final_results["general_after"]

    forgetting=(
        general_after["loss"]
        -
        general_before["loss"]
    )

    result={
        "experiment":"A2",
        "node":"global",
        "replay_ratio":rho,
        "seed":seed,

        "rounds":ROUNDS,

        "general_loss_before":
            general_before["loss"],

        "general_loss_after":
            general_after["loss"],

        "forgetting":forgetting,

        "general_loss_change_pct":
            100*forgetting/general_before["loss"]
            if general_before["loss"]!=0
            else None,

        "general_ppl_before":
            general_before["perplexity"],

        "general_ppl_after":
            general_after["perplexity"],

        "general_eval_tokens":
            general_after["tokens"],

        "replay_pool_tokens":
            replay_pool_tokens,

        "training_seconds":
            training_seconds,

        "model":MODEL_PATH,

        "max_length":MAX_LENGTH,

        "batch_size":BATCH_SIZE,

        "learning_rate":LR,

        "local_steps":LOCAL_STEPS,

        "training_examples_per_node_per_round":
            TRAINING_EXAMPLES_PER_NODE_PER_ROUND
    }

    # --------------------------------------------------------
    # PER-NODE DOMAIN RESULTS
    # --------------------------------------------------------

    for node in NODES:

        before=base_results["domains"][node]
        after=final_results["domains"][node]

        domain_improvement=(
            before["loss"]
            -
            after["loss"]
        )

        result[f"node{node}_domain_loss_before"]={
            "value":before["loss"]
        }

        result[f"node{node}_domain_loss_after"]={
            "value":after["loss"]
        }

        result[f"node{node}_domain_improvement"]={
            "value":domain_improvement
        }

        result[f"node{node}_domain_loss_change_pct"]={
            "value":
                100*(after["loss"]-before["loss"])
                /before["loss"]
                if before["loss"]!=0
                else None
        }

        result[f"node{node}_domain_ppl_before"]={
            "value":before["perplexity"]
        }

        result[f"node{node}_domain_ppl_after"]={
            "value":after["perplexity"]
        }

        result[f"node{node}_domain_eval_tokens"]={
            "value":after["tokens"]
        }

    return result


# ============================================================
# FLATTEN RESULT FOR CSV
# ============================================================

def flatten_result(result):

    flat={}

    for key,value in result.items():

        if isinstance(value,dict):

            if "value" in value and len(value)==1:
                flat[key]=value["value"]

            else:
                for subkey,subvalue in value.items():

                    if isinstance(subvalue,dict):
                        continue

                    flat[f"{key}_{subkey}"]=subvalue

        else:
            flat[key]=value

    return flat


# ============================================================
# SAVE RESULTS
# ============================================================

def save_all_results(results):

    import csv

    if not results:
        return

    json_path=os.path.join(
        RESULTS_DIR,
        "experimentA2_all_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    csv_path=os.path.join(
        RESULTS_DIR,
        "all_results.csv"
    )

    flat_results=[
        flatten_result(r)
        for r in results
    ]

    keys=[]

    for result in flat_results:

        for key in result:

            if key not in keys:
                keys.append(key)

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer=csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()
        writer.writerows(flat_results)

    print(
        f"\nSaved A2 JSON: {json_path}"
    )

    print(
        f"Saved A2 CSV:  {csv_path}"
    )


# ============================================================
# SIMPLE A2 SUMMARY
# ============================================================

def summarize_results(results):

    print("\n")
    print("============================================================")
    print("                 EXPERIMENT A2 SUMMARY")
    print("============================================================")

    for rho in RHO_GRID:

        rows=[
            r for r in results
            if r["replay_ratio"]==rho
        ]

        if not rows:
            continue

        avg_forgetting=sum(
            r["forgetting"]
            for r in rows
        )/len(rows)

        print(
            f"\nGlobal rho={rho:.2f}"
        )

        print(
            f"  Mean general forgetting: "
            f"{avg_forgetting:+.6f}"
        )

        for node in NODES:

            values=[
                r[
                    f"node{node}_domain_improvement"
                ]["value"]
                for r in rows
            ]

            avg_improvement=sum(values)/len(values)

            print(
                f"  Node {node} mean domain improvement: "
                f"{avg_improvement:+.6f}"
            )

    print("\n============================================================")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment_a2(only_rho=None,only_seed=None):

    print("\n")
    print("============================================================")
    print("              EXPERIMENT A2")
    print("       FEDERATED GLOBAL FIXED REPLAY")
    print("============================================================")

    print("NodeReplay:        DISABLED")
    print(f"Model:             {MODEL_PATH}")
    print(f"Nodes:              {NODES}")
    print(f"Replay ratios:      {RHO_GRID}")
    print(f"Seeds:              {SEEDS}")
    print(f"Rounds:             {ROUNDS}")

    print(
        f"Total runs:         "
        f"{len(RHO_GRID)*len(SEEDS)}"
    )

    print(
        f"Local samples:      "
        f"{LOCAL_SAMPLES}"
    )

    print(
        f"Replay samples:     "
        f"{REPLAY_SAMPLES}"
    )

    print(
        f"Evaluation samples: "
        f"{EVAL_SAMPLES}"
    )

    print(
        f"Local steps:        "
        f"{LOCAL_STEPS}"
    )

    print(
        f"Batch size:         "
        f"{BATCH_SIZE}"
    )

    print(
        f"Training examples/node/round: "
        f"{TRAINING_EXAMPLES_PER_NODE_PER_ROUND}"
    )

    print(
        f"Max length:         "
        f"{MAX_LENGTH}"
    )

    print(
        f"Learning rate:      "
        f"{LR}"
    )

    print(
        f"Device:             "
        f"{DEVICE}"
    )

    print("============================================================\n")

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            RESULTS_DIR,
            "runs"
        ),
        exist_ok=True
    )

    config_path=save_config()

    print(
        f"Saved configuration: "
        f"{config_path}"
    )

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    tokenizer=get_tokenizer()

    # ========================================================
    # LOAD GENERAL REPLAY POOL
    # ========================================================

    print("\nLoading general replay pool...")

    replay_data=get_replay_dataset(
        max_samples=REPLAY_SAMPLES,
        start_index=0
    )

    print(
        f"Replay examples loaded: "
        f"{len(replay_data)}"
    )

    replay_pool_tokens=count_tokens(
        replay_data,
        tokenizer,
        MAX_LENGTH
    )

    print(
        f"Replay pool tokens: "
        f"{replay_pool_tokens}"
    )

    # ========================================================
    # LOAD HELD-OUT GENERAL EVALUATION DATA
    # ========================================================

    print("\nLoading held-out general evaluation data...")

    general_eval=get_replay_dataset(
        max_samples=EVAL_SAMPLES,
        start_index=REPLAY_SAMPLES
    )

    print(
        f"General evaluation examples: "
        f"{len(general_eval)}"
    )

    general_eval_tokens=count_tokens(
        general_eval,
        tokenizer,
        MAX_LENGTH
    )

    print(
        f"General evaluation tokens: "
        f"{general_eval_tokens}"
    )

    # ========================================================
    # LOAD ALL DOMAIN DATA
    # ========================================================

    domain_train_data={}
    domain_eval_data={}
    domain_eval_tokens={}

    for node in NODES:

        print("\n")
        print("------------------------------------------------------------")
        print(f"Loading data for Node {node}")
        print("------------------------------------------------------------")

        domain_train_data[node]=get_dataset(
            node=node,
            max_samples=LOCAL_SAMPLES,
            start_index=0
        )

        domain_eval_data[node]=get_dataset(
            node=node,
            max_samples=EVAL_SAMPLES,
            start_index=LOCAL_SAMPLES
        )

        domain_eval_tokens[node]=count_tokens(
            domain_eval_data[node],
            tokenizer,
            MAX_LENGTH
        )

        print(
            f"Node {node} training examples: "
            f"{len(domain_train_data[node])}"
        )

        print(
            f"Node {node} evaluation examples: "
            f"{len(domain_eval_data[node])}"
        )

        print(
            f"Node {node} evaluation tokens: "
            f"{domain_eval_tokens[node]}"
        )

    # ========================================================
    # BASE MODEL EVALUATION
    # ========================================================

    base_results=evaluate_base_model(
        general_eval,
        domain_eval_data
    )

    # ========================================================
    # EXPERIMENT RESULTS
    # ========================================================

    results=[]

    expected_runs=(
        len(RHO_GRID)
        *
        len(SEEDS)
    )

    # ========================================================
    # GLOBAL REPLAY RATIO LOOP
    # ========================================================

    for rho in RHO_GRID:
        if only_rho is not None and rho!=only_rho:
            continue

        print("\n")
        print("############################################################")
        print(
            f"                 GLOBAL RHO = {rho:.2f}"
        )
        print("############################################################")

        # ====================================================
        # SEED LOOP
        # ====================================================

        for seed in SEEDS:
            if only_seed is not None and seed!=only_seed:
                continue

            run_name=get_run_name(
                rho,
                seed
            )

            run_results_dir=os.path.join(
                RESULTS_DIR,
                "runs",
                run_name
            )

            # ------------------------------------------------
            # RESUME SUPPORT
            # ------------------------------------------------

            if run_already_completed(
                rho,
                seed
            ):

                print(
                    f"\nSkipping completed run: "
                    f"{run_name}"
                )

                result_path=os.path.join(
                    run_results_dir,
                    "result.json"
                )

                with open(
                    result_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    completed_result=json.load(f)

                results.append(
                    completed_result
                )

                continue

            print("\n")
            print("============================================================")
            print(
                f"RUN | global rho={rho:.2f} | "
                f"seed={seed}"
            )
            print("============================================================")

            set_all_seeds(seed)

            # ------------------------------------------------
            # TRAIN FEDERATED MODEL
            # ------------------------------------------------
            check_disk_space(RESULTS_DIR)

            print("\nStarting federated training...")

            start_time=time.time()

            global_model=run_federated(
                nodes=NODES,
                rounds=ROUNDS,
                local_steps=LOCAL_STEPS,
                batch_size=BATCH_SIZE,
                replay_ratio=rho,
                max_local_samples=LOCAL_SAMPLES,
                max_replay_samples=REPLAY_SAMPLES,
                seed=seed
            )

            training_seconds=(
                time.time()-start_time
            )

            print(
                f"\nFederated training complete: "
                f"{training_seconds:.2f}s"
            )

            # ------------------------------------------------
            # SAVE FINAL GLOBAL CHECKPOINT
            # ------------------------------------------------

            if SAVE_CHECKPOINTS:
                check_disk_space(CHECKPOINT_DIR,min_free_gb=MIN_FREE_GB+10)
                checkpoint_path=os.path.join(CHECKPOINT_DIR,run_name)
                save_checkpoint(global_model,checkpoint_path)

            # ------------------------------------------------
            # EVALUATE FINAL GLOBAL MODEL
            # ------------------------------------------------

            print("\nRunning global evaluation...")

            final_tokenizer=get_tokenizer()

            final_results=evaluate_global_model(
                global_model,
                final_tokenizer,
                DEVICE,
                general_eval,
                domain_eval_data
            )

            # ------------------------------------------------
            # BUILD RESULT
            # ------------------------------------------------

            result=build_result(
                rho=rho,
                seed=seed,
                base_results=base_results,
                final_results=final_results,
                training_seconds=training_seconds,
                replay_pool_tokens=replay_pool_tokens,
                general_eval_tokens=general_eval_tokens,
                domain_eval_tokens=domain_eval_tokens
            )

            # ------------------------------------------------
            # SAVE INDIVIDUAL RESULT
            # ------------------------------------------------

            os.makedirs(
                run_results_dir,
                exist_ok=True
            )

            result_path=os.path.join(
                run_results_dir,
                "result.json"
            )

            with open(
                result_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=2
                )

            print(
                f"\nSaved run result:"
                f"\n{result_path}"
            )

            results.append(result)

            # ------------------------------------------------
            # CLEANUP MODEL
            # ------------------------------------------------

            del global_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            print("\nRun complete.")

            print(
                f"Progress: "
                f"{len(results)}/{expected_runs}"
            )

    # ========================================================
    # SAVE COMPLETE RESULT TABLE
    # ========================================================

    print("\n")
    print("============================================================")
    print("Saving complete Experiment A2 dataset...")
    print("============================================================")

    if only_rho is None and only_seed is None:
        save_all_results(results)
        summarize_results(results)
    else:
        print("Single-run mode: skipping aggregate CSV. "
              "Run with --aggregate once all jobs finish.")

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("============================================================")
    print("             EXPERIMENT A2 COMPLETE")
    print("============================================================")

    print(
        f"Completed runs: "
        f"{len(results)}/{expected_runs}"
    )

    print(
        f"Global replay ratios: "
        f"{len(RHO_GRID)}"
    )

    print(
        f"Seeds: "
        f"{len(SEEDS)}"
    )

    print(
        f"Federated rounds/run: "
        f"{ROUNDS}"
    )

    print("\nRecorded:")

    print(
        "  - General loss before/after"
    )

    print(
        "  - General forgetting"
    )

    print(
        "  - General perplexity"
    )

    for node in NODES:

        print(
            f"  - Node {node} domain loss "
            f"before/after"
        )

        print(
            f"  - Node {node} domain improvement"
        )

        print(
            f"  - Node {node} domain perplexity"
        )

    print(
        "  - Replay pool tokens"
    )

    print(
        "  - Training time"
    )

    print(
        "  - Seed"
    )

    print(
        "  - Global replay ratio"
    )

    print(
        f"\nResults: "
        f"{RESULTS_DIR}"
    )

    print(
        f"Checkpoints: "
        f"{CHECKPOINT_DIR}"
    )

    print("============================================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--rho",type=float,default=None)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args()

    if args.aggregate:
        aggregate_results()
    else:
        run_experiment_a2(only_rho=args.rho,only_seed=args.seed)