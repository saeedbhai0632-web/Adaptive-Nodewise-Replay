
import os
import time
import json
import random
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed

from src.data import get_dataset
from src.replay import get_replay_dataset
from src.mixer import get_mixed_dataset
from src.trainer import train
from src.evaluator import evaluate_condition,save_results,plot_experiment_a,summarize_experiment_a


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH="./pretrained/Qwen2.5-1.5B"

NODES=[1,2,3,4]

RHO_GRID=[0.00,0.05,0.10,0.20,0.40]

SEEDS=[42,43,44]

LOCAL_SAMPLES=1000

REPLAY_SAMPLES=1000

EVAL_SAMPLES=500

LOCAL_STEPS=100

BATCH_SIZE=2

MAX_LENGTH=512

LR=1e-5
TRAINING_EXAMPLES=LOCAL_STEPS*BATCH_SIZE

RESULTS_DIR="./experiments/results/experimentA"
CHECKPOINT_DIR="./experiments/checkpoints/experimentA"

DEVICE="cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

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


def count_mixed_tokens(mixed_data,tokenizer,max_length=512):
    local_tokens=0
    replay_tokens=0

    for i in range(0,len(mixed_data),32):
        batch=mixed_data[i:i+32]

        texts=[x[1] for x in batch]

        inputs=tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )

        for j,ids in enumerate(inputs["input_ids"]):
            if batch[j][0]=="local":
                local_tokens+=len(ids)
            else:
                replay_tokens+=len(ids)

    return local_tokens,replay_tokens


def save_checkpoint(model,checkpoint_path):
    os.makedirs(checkpoint_path,exist_ok=True)

    model.save_pretrained(checkpoint_path)

    tokenizer=get_tokenizer()
    tokenizer.save_pretrained(checkpoint_path)


def save_config():
    config={
        "experiment":"Experiment A",
        "description":"Fixed replay baseline without NodeReplay",
        "model":MODEL_PATH,
        "nodes":NODES,
        "replay_ratios":RHO_GRID,
        "seeds":SEEDS,
        "local_samples":LOCAL_SAMPLES,
        "replay_samples":REPLAY_SAMPLES,
        "eval_samples":EVAL_SAMPLES,
        "local_steps":LOCAL_STEPS,
        "batch_size":BATCH_SIZE,
        "training_examples":TRAINING_EXAMPLES,
        "max_length":MAX_LENGTH,
        "learning_rate":LR,
        "device":DEVICE
    }

    path=os.path.join(
        RESULTS_DIR,
        "experimentA_config.json"
    )

    with open(path,"w",encoding="utf-8") as f:
        json.dump(config,f,indent=2)

    return path


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_already_completed(node,rho,seed):
    rho_name=f"{rho:.2f}".replace(".","")

    run_results_dir=os.path.join(
        RESULTS_DIR,
        "runs",
        f"node{node}_rho{rho_name}_seed{seed}"
    )

    if not os.path.exists(run_results_dir):
        return False

    files=os.listdir(run_results_dir)

    return len(files)>0
def run_experiment_a():

    print("\n")
    print("============================================================")
    print("                 EXPERIMENT A")
    print("              FIXED REPLAY BASELINE")
    print("============================================================")
    print("NodeReplay:        DISABLED")
    print(f"Model:             {MODEL_PATH}")
    print(f"Nodes:             {NODES}")
    print(f"Replay ratios:     {RHO_GRID}")
    print(f"Seeds:             {SEEDS}")
    print(
        f"Total runs:        "
        f"{len(NODES)*len(RHO_GRID)*len(SEEDS)}"
    )
    print(f"Local samples:     {LOCAL_SAMPLES}")
    print(f"Replay samples:    {REPLAY_SAMPLES}")
    print(f"Evaluation samples:{EVAL_SAMPLES}")
    print(f"Local steps:       {LOCAL_STEPS}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Training examples: {TRAINING_EXAMPLES}")
    print(f"Max length:        {MAX_LENGTH}")
    print(f"Learning rate:     {LR}")
    print(f"Device:            {DEVICE}")
    print("============================================================\n")

    os.makedirs(RESULTS_DIR,exist_ok=True)
    os.makedirs(CHECKPOINT_DIR,exist_ok=True)

    config_path=save_config()

    print(f"Saved configuration: {config_path}")

    tokenizer=get_tokenizer()

    # --------------------------------------------------------
    # LOAD GENERAL REPLAY DATA
    # --------------------------------------------------------

    print("\nLoading general replay data...")

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

    # --------------------------------------------------------
    # LOAD HELD-OUT GENERAL EVALUATION DATA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # STORE ALL RESULTS
    # --------------------------------------------------------

    results=[]

    # ========================================================
    # NODE LOOP
    # ========================================================

    for node in NODES:

        print("\n")
        print("############################################################")
        print(f"                         NODE {node}")
        print("############################################################")

        # ----------------------------------------------------
        # LOCAL TRAINING DATA
        # ----------------------------------------------------

        print("\nLoading local training data...")

        local_data=get_dataset(
            node=node,
            max_samples=LOCAL_SAMPLES,
            start_index=0
        )

        print(
            f"Local training examples: "
            f"{len(local_data)}"
        )

        # ----------------------------------------------------
        # LOCAL HELD-OUT EVALUATION DATA
        # ----------------------------------------------------

        print("Loading local evaluation data...")

        domain_eval=get_dataset(
            node=node,
            max_samples=EVAL_SAMPLES,
            start_index=LOCAL_SAMPLES
        )

        print(
            f"Local evaluation examples: "
            f"{len(domain_eval)}"
        )

        local_eval_tokens=count_tokens(
            domain_eval,
            tokenizer,
            MAX_LENGTH
        )

        print(
            f"Local evaluation tokens: "
            f"{local_eval_tokens}"
        )

        # ====================================================
        # REPLAY RATIO LOOP
        # ====================================================

        for rho in RHO_GRID:

            print("\n")
            print("------------------------------------------------------------")
            print(f"NODE {node} | REPLAY RATIO {rho}")
            print("------------------------------------------------------------")

            # =================================================
            # SEED LOOP
            # =================================================

            for seed in SEEDS:
#the next if condition is the addition to resume from where we left

                if run_already_completed(node,rho,seed):
                    print(
                        f"Skipping completed run: "
                        f"node={node} | rho={rho} | seed={seed}"
                    )
                    continue

                print("\n")
                print("============================================================")
                print(
                    f"RUN | node={node} | "
                    f"rho={rho} | seed={seed}"
                )
                print("============================================================")

                set_all_seeds(seed)

                # ------------------------------------------------
                # CREATE EXACT TRAINING MIXTURE
                # ------------------------------------------------

                mixed_data=get_mixed_dataset(
                    local_data=local_data,
                    replay_data=replay_data,
                    replay_ratio=rho,
                    total_examples=TRAINING_EXAMPLES,
                    seed=seed
                )

                # ------------------------------------------------
                # COUNT ACTUAL TRAINING TOKENS
                # ------------------------------------------------

                local_tokens,replay_tokens=count_mixed_tokens(
                    mixed_data,
                    tokenizer,
                    MAX_LENGTH
                )

                total_tokens=local_tokens+replay_tokens

                actual_replay_fraction=(
                    replay_tokens/total_tokens
                    if total_tokens>0
                    else 0
                )

                print(
                    f"Mixed examples:       "
                    f"{len(mixed_data)}"
                )

                print(
                    f"Local tokens:          "
                    f"{local_tokens}"
                )

                print(
                    f"Replay tokens:         "
                    f"{replay_tokens}"
                )

                print(
                    f"Total tokens:          "
                    f"{total_tokens}"
                )

                print(
                    f"Actual replay frac:    "
                    f"{actual_replay_fraction:.4f}"
                )

                # ------------------------------------------------
                # LOAD FRESH THETA_0
                # ------------------------------------------------

                print("\nLoading theta_0...")

                model=AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH
                )

                # ------------------------------------------------
                # TRAIN
                # ------------------------------------------------

                print("Starting training...")

                start_time=time.time()

                model=train(
                    model=model,
                    mixed_data=mixed_data,
                    batch_size=BATCH_SIZE,
                    max_length=MAX_LENGTH,
                    steps=LOCAL_STEPS,
                    lr=LR
                )

                training_seconds=time.time()-start_time

                print(
                    f"Training complete: "
                    f"{training_seconds:.2f}s"
                )

                # ------------------------------------------------
                # SAVE CHECKPOINT
                # ------------------------------------------------

                rho_name=f"{rho:.2f}".replace(".","")

                checkpoint_path=os.path.join(
                    CHECKPOINT_DIR,
                    f"node{node}_rho{rho_name}_seed{seed}"
                )

                print(
                    f"Saving checkpoint: "
                    f"{checkpoint_path}"
                )

                save_checkpoint(
                    model,
                    checkpoint_path
                )

                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ------------------------------------------------
                # RUN EVALUATION
                # ------------------------------------------------

                print("Running evaluation...")

                run_results_dir=os.path.join(
                    RESULTS_DIR,
                    "runs",
                    f"node{node}_rho{rho_name}_seed{seed}"
                )

                os.makedirs(
                    run_results_dir,
                    exist_ok=True
                )

                result=evaluate_condition(
                    base_model_path=MODEL_PATH,
                    trained_model_path=checkpoint_path,
                    general_data=general_eval,
                    domain_data=domain_eval,
                    node=node,
                    replay_ratio=rho,
                    replay_tokens=replay_tokens,
                    training_seconds=training_seconds,
                    output_dir=run_results_dir,
                    max_length=MAX_LENGTH,
                    batch_size=BATCH_SIZE
                )
                import shutil
                shutil.rmtree(checkpoint_path,ignore_errors=True)
                # ------------------------------------------------
                # ADD EXPERIMENT METADATA
                # ------------------------------------------------

                result["seed"]=seed
                result["local_tokens"]=local_tokens
                result["replay_tokens"]=replay_tokens
                result["total_training_tokens"]=total_tokens
                result["actual_replay_fraction"]=actual_replay_fraction
                result["local_eval_tokens"]=local_eval_tokens
                result["general_eval_tokens"]=general_eval_tokens
                result["model"]=MODEL_PATH
                result["max_length"]=MAX_LENGTH
                result["batch_size"]=BATCH_SIZE
                result["learning_rate"]=LR
                result["local_steps"]=LOCAL_STEPS
                result["training_examples"]=TRAINING_EXAMPLES

                results.append(result)

                print("\nRun complete.")

                print(
                    f"Progress: "
                    f"{len(results)}/"
                    f"{len(NODES)*len(RHO_GRID)*len(SEEDS)}"
                )

    # ========================================================
    # SAVE COMPLETE RESULT TABLE
    # ========================================================

    print("\n")
    print("============================================================")
    print("Saving complete Experiment A dataset...")
    print("============================================================")

    save_results(
        results,
        output_dir=RESULTS_DIR
    )

    # ========================================================
    # SAVE RAW RESULTS AS JSON
    # ========================================================

    raw_results_path=os.path.join(
        RESULTS_DIR,
        "experimentA_all_results.json"
    )

    with open(
        raw_results_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"Raw results saved to: "
        f"{raw_results_path}"
    )

    # ========================================================
    # PLOTS
    # ========================================================

    print("\n")
    print("============================================================")
    print("Generating Experiment A plots...")
    print("============================================================")

    plot_experiment_a(
        results,
        output_dir=os.path.join(
            RESULTS_DIR,
            "plots"
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summarize_experiment_a(results)

    # ========================================================
    # FINAL STATUS
    # ========================================================

    expected_runs=(
        len(NODES)
        *len(RHO_GRID)
        *len(SEEDS)
    )

    print("\n")
    print("============================================================")
    print("             EXPERIMENT A COMPLETE")
    print("============================================================")
    print("Baseline:          FIXED REPLAY")
    print("NodeReplay:        NOT USED")
    print(f"Completed runs:    {len(results)}/{expected_runs}")
    print(f"Nodes:             {len(NODES)}")
    print(f"Replay ratios:     {len(RHO_GRID)}")
    print(f"Seeds:             {len(SEEDS)}")
    print("")
    print("Recorded:")
    print("  - General loss before/after")
    print("  - General forgetting")
    print("  - Domain loss before/after")
    print("  - Domain improvement")
    print("  - Replay tokens")
    print("  - Local tokens")
    print("  - Total training tokens")
    print("  - Actual replay fraction")
    print("  - Training time")
    print("  - General PPL")
    print("  - Domain PPL")
    print("  - Seed")
    print("")
    print(f"Results: {RESULTS_DIR}")
    print(
        f"Plots:   "
        f"{os.path.join(RESULTS_DIR,'plots')}"
    )
    print("============================================================")


if __name__=="__main__":
    run_experiment_a()

