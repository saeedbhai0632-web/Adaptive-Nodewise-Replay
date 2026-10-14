"""
Experiment B: NodeReplay (per-node adaptive rho) vs fixed-ratio baselines,
all under the SAME federated protocol (same data slices, rounds, local steps,
seeds, FedAvg), so the comparison is fair.

Workflow
--------
1) Calibrate (no adaptation, only measures per-node F and P signals):
    CUDA_VISIBLE_DEVICES=2 python -u run_experimentB.py --mode calibrate --seed 42
   (optionally repeat with --init_rho 0.0 / 0.05 / 0.20 / 0.40 to see how F reacts to rho)

2) Pick thresholds (validation signals only, never the final-eval slices):
    python run_experimentB.py --mode suggest
   -> prints the signal table and a suggested --tau_f/--tau_low/--tau_s

3) Baselines, same protocol (no replay, 5/10/20/40 %):
    for r in 0.0 0.05 0.10 0.20 0.40; do
      CUDA_VISIBLE_DEVICES=2 python -u run_experimentB.py --mode fixed --init_rho $r --seed 42
    done

4) NodeReplay:
    CUDA_VISIBLE_DEVICES=2 python -u run_experimentB.py --mode run --seed 42 \
        --tau_f X --tau_low Y --tau_s Z

5) Per-node oracle (rho per node chosen from Experiment A's per-node curves):
    python -u run_experimentB.py --mode fixed --node_rhos 0.1,0.2,0.0,0.4 --seed 42

6) Repeat 3-4 for seeds 42, 43, 44, then:
    python run_experimentB.py --mode summarize
"""
import argparse
import glob
import json
import os
import statistics
import time

import run_experimentA2 as A2
from src.data import get_dataset
from src.replay import get_replay_dataset
from src.nodereplay import run_nodereplay, RHO_GRID

RESULTS_DIR="./experiments/results/experimentB" 
A2_RESULTS_DIR="./experiments/results/experimentA2"

ANCHOR_SAMPLES=100
VAL_SAMPLES=100


# Controller data lives AFTER the final-evaluation slices used in A2
# (train 0-999, final eval 1000-1499, controller data 1500+),
# so the controller never sees the data it is finally judged on.
GENERAL_ANCHOR_START=A2.REPLAY_SAMPLES+A2.EVAL_SAMPLES
DOMAIN_VAL_START=A2.LOCAL_SAMPLES+A2.EVAL_SAMPLES


def rho_tag(rho):
    return f"{rho:.2f}".replace(".","")


def load_control_data():
    print("\nLoading controller data (anchor + per-node validation)...")

    anchor=get_replay_dataset(
        max_samples=ANCHOR_SAMPLES,
        start_index=GENERAL_ANCHOR_START
    )

    domain_val={}
    for node in A2.NODES:
        domain_val[node]=get_dataset(
            node=node,
            max_samples=VAL_SAMPLES,
            start_index=DOMAIN_VAL_START
        )

    print(f"Anchor examples: {len(anchor)}")
    for node in A2.NODES:
        print(f"Node {node} validation examples: {len(domain_val[node])}")

    return anchor,domain_val


def load_final_eval_data():
    # Identical slices to A2, so results are directly comparable.
    general_eval=get_replay_dataset(
        max_samples=A2.EVAL_SAMPLES,
        start_index=A2.REPLAY_SAMPLES
    )

    domain_eval={}
    for node in A2.NODES:
        domain_eval[node]=get_dataset(
            node=node,
            max_samples=A2.EVAL_SAMPLES,
            start_index=A2.LOCAL_SAMPLES
        )

    return general_eval,domain_eval


def print_history(history):
    print("\n")
    print("============================================================")
    print("   PER-NODE SIGNALS (F = forgetting, P = improvement)")
    print("============================================================")

    for h in history:
        print(
            f"\nRound {h['round']} | global forgetting "
            f"{h['global_forgetting']:+.5f}"
        )
        for rec in h["nodes"]:
            print(
                f"  Node {rec['node']} | rho {rec['rho']:.2f}"
                f" -> {rec['next_rho']:.2f}"
                f" | F {rec['forgetting']:+.5f}"
                f" | P {rec['improvement']:+.5f}"
                f" | replay ex {rec['replay_examples']}"
            )


def parse_node_rhos(text):
    vals=[float(x) for x in text.split(",")]
    if len(vals)!=len(A2.NODES):
        raise SystemExit(f"--node_rhos needs {len(A2.NODES)} values (one per node {A2.NODES})")
    for v in vals:
        if v not in RHO_GRID:
            raise SystemExit(f"--node_rhos values must be in {RHO_GRID}")
    return dict(zip(A2.NODES,vals))


# ----------------------------------------------------------------------
# calibrate
# ----------------------------------------------------------------------
def calibrate(args):
    os.makedirs(os.path.join(RESULTS_DIR,"calibration"),exist_ok=True)
    A2.check_disk_space(RESULTS_DIR)

    anchor,domain_val=load_control_data()
    A2.set_all_seeds(args.seed)

    path=os.path.join(
        RESULTS_DIR,"calibration",
        f"calib_rho{rho_tag(args.init_rho)}_seed{args.seed}.json"
    )

    _,history=run_nodereplay(
        nodes=A2.NODES,
        rounds=args.rounds,
        local_steps=args.local_steps,
        batch_size=A2.BATCH_SIZE,
        init_rho=args.init_rho,
        max_local_samples=A2.LOCAL_SAMPLES,
        max_replay_samples=A2.REPLAY_SAMPLES,
        anchor_data=anchor,
        domain_val=domain_val,
        adaptive=False,
        seed=args.seed,
        history_path=path
    )

    print_history(history)
    print(f"\nSaved calibration history: {path}")


# ----------------------------------------------------------------------
# suggest thresholds from calibration files
# ----------------------------------------------------------------------
def _quantile(vals,q):
    vals=sorted(vals)
    pos=q*(len(vals)-1)
    lo=int(pos)
    hi=min(lo+1,len(vals)-1)
    return vals[lo]+(vals[hi]-vals[lo])*(pos-lo)


def suggest(args):
    files=sorted(glob.glob(os.path.join(RESULTS_DIR,"calibration","calib_rho*_seed*.json")))
    if not files:
        raise SystemExit("No calibration files found. Run --mode calibrate first.")

    print("============================================================")
    print("  CALIBRATION SIGNALS  (F = forgetting, P = improvement)")
    print("============================================================")

    ref_F=[]
    ref_P=[]

    for path in files:
        with open(path,encoding="utf-8") as f:
            history=json.load(f)

        rho0=history[0]["nodes"][0]["rho"]
        print(f"\n{os.path.basename(path)}  (fixed rho={rho0:.2f})")

        for h in history:
            for rec in h["nodes"]:
                print(
                    f"  round {h['round']} node {rec['node']} | "
                    f"F {rec['forgetting']:+.5f} | P {rec['improvement']:+.5f}"
                )
                if abs(rho0-args.init_rho)<1e-9:
                    ref_F.append(rec["forgetting"])
                    ref_P.append(rec["improvement"])

    if len(ref_F)<4:
        raise SystemExit(
            f"Need calibration at --init_rho {args.init_rho} "
            f"(found {len(ref_F)} node-round samples) to suggest thresholds."
        )

    # Heuristic starting point. Eyeball the table above and override if needed.
    #   tau_f   = median F      -> roughly the more-forgetful half of node-rounds push rho up
    #   tau_low = 25th pct of F -> only clearly safe node-rounds are allowed to step down
    #   tau_s   = 25th pct of P -> "slow specialization" = bottom quarter of improvements
    tau_f=_quantile(ref_F,0.50)
    tau_low=_quantile(ref_F,0.25)
    tau_s=_quantile(ref_P,0.25)

    print("\n============================================================")
    print(f"Suggested thresholds (from rho={args.init_rho:.2f} calibration, {len(ref_F)} samples):")
    print(f"  tau_f={tau_f:.5f}  tau_low={tau_low:.5f}  tau_s={tau_s:.5f}")
    print("Sanity checks before using them:")
    print("  - tau_low must be < tau_f (hysteresis band)")
    print("  - if every node has F >> tau_f at rho=0, rho will just ramp up to 0.40 (no control)")
    print("  - if all F are ~0, replay is not needed at this scale: raise LR/steps instead")
    print("============================================================")
    print(
        f"\nCUDA_VISIBLE_DEVICES=2 python -u run_experimentB.py --mode run --seed {args.seed} "
        f"--tau_f {tau_f:.5f} --tau_low {tau_low:.5f} --tau_s {tau_s:.5f}"
    )


# ----------------------------------------------------------------------
# run (adaptive NodeReplay) and fixed baselines share one code path
# ----------------------------------------------------------------------
def execute(args,adaptive):
    if adaptive:
        for name in ("tau_f","tau_low","tau_s"):
            if getattr(args,name) is None:
                raise SystemExit(f"--{name} is required in run mode (use --mode suggest first)")
        if not args.tau_low<args.tau_f:
            raise SystemExit("--tau_low must be smaller than --tau_f")
        method="nodereplay"
        init_rho=args.init_rho
        run_name=f"nodereplay_init{rho_tag(args.init_rho)}_seed{args.seed}{args.tag}"
        controller_cfg={
            "tau_f":args.tau_f,
            "tau_low":args.tau_low,
            "tau_s":args.tau_s
        }
    else:
        method="fixed"
        controller_cfg=None
        if args.node_rhos:
            init_rho=parse_node_rhos(args.node_rhos)
            label="oracle_"+"-".join(rho_tag(init_rho[n]) for n in A2.NODES)
        else:
            init_rho=args.init_rho
            label=f"rho{rho_tag(args.init_rho)}"
        run_name=f"fixed_{label}_seed{args.seed}{args.tag}"

    run_dir=os.path.join(RESULTS_DIR,"runs",run_name)
    result_path=os.path.join(run_dir,"result.json")
    history_path=os.path.join(run_dir,"history.json")

    if os.path.exists(result_path):
        print(f"Run already completed, skipping: {run_name}")
        return

    os.makedirs(run_dir,exist_ok=True)
    A2.check_disk_space(RESULTS_DIR)

    print("\n============================================================")
    print(f"{method.upper()} | {run_name}")
    print(f"init rho={init_rho} | thresholds={controller_cfg}")
    print(f"rounds={args.rounds} | local steps={args.local_steps}")
    print("============================================================")

    anchor,domain_val=load_control_data()
    general_eval,domain_eval=load_final_eval_data()

    base_results=A2.evaluate_base_model(general_eval,domain_eval)

    A2.set_all_seeds(args.seed)
    start=time.time()

    global_model,history=run_nodereplay(
        nodes=A2.NODES,
        rounds=args.rounds,
        local_steps=args.local_steps,
        batch_size=A2.BATCH_SIZE,
        init_rho=init_rho,
        max_local_samples=A2.LOCAL_SAMPLES,
        max_replay_samples=A2.REPLAY_SAMPLES,
        anchor_data=anchor,
        domain_val=domain_val,
        controller_cfg=controller_cfg,
        adaptive=adaptive,
        seed=args.seed,
        history_path=history_path
    )

    training_seconds=time.time()-start

    final_results=A2.evaluate_global_model(
        global_model,
        A2.get_tokenizer(),
        A2.DEVICE,
        general_eval,
        domain_eval
    )

    result=A2.build_result(
        rho=None,
        seed=args.seed,
        base_results=base_results,
        final_results=final_results,
        training_seconds=training_seconds,
        replay_pool_tokens=None,
        general_eval_tokens=None,
        domain_eval_tokens=None
    )

    all_rhos=[rec["rho"] for h in history for rec in h["nodes"]]

    result["experiment"]="B"
    result["method"]=method
    result["run_name"]=run_name
    result["replay_ratio"]=None
    result["rounds"]=args.rounds
    result["local_steps"]=args.local_steps
    result["init_rho"]=init_rho if not isinstance(init_rho,dict) else {str(k):v for k,v in init_rho.items()}
    result["mean_rho"]=sum(all_rhos)/len(all_rhos)
    result["total_replay_examples"]=sum(
        rec["replay_examples"] for h in history for rec in h["nodes"]
    )
    result["total_replay_tokens"]=sum(
        rec["replay_tokens"] for h in history for rec in h["nodes"]
    )
    result["controller"]=controller_cfg
    result["rho_trajectory"]={
        str(node):[
            rec["rho"]
            for h in history for rec in h["nodes"] if rec["node"]==node
        ]
        for node in A2.NODES
    }
    # Validation-slice signals at the final round: used to pick the "best global
    # fixed" ratio WITHOUT touching the final-eval slices.
    last=history[-1]
    result["val_global_forgetting"]=last["global_forgetting"]
    result["val_mean_domain_loss"]=sum(last["global_domain_loss"].values())/len(last["global_domain_loss"])

    with open(result_path,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)

    print_history(history)

    print("\n============================================================")
    print(f"{method.upper()} RESULT")
    print("============================================================")
    print(f"General forgetting:    {result['forgetting']:+.6f}")
    for node in A2.NODES:
        print(
            f"Node {node} improvement:   "
            f"{result[f'node{node}_domain_improvement']['value']:+.6f}"
        )
    print(f"Mean rho:              {result['mean_rho']:.4f}")
    print(f"Total replay examples: {result['total_replay_examples']}")
    print(f"Total replay tokens:   {result['total_replay_tokens']}")
    print(f"\nSaved: {result_path}")


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------
def _group_key(r):
    if r.get("experiment")=="A2":
        return f"A2_fixed_rho{rho_tag(r['replay_ratio'])}"
    if r.get("method")=="nodereplay":
        return f"nodereplay(init={r['init_rho']})"
    return r.get("run_name","fixed").split("_seed")[0]


def _load(pattern):
    out=[]
    for p in sorted(glob.glob(pattern)):
        with open(p,encoding="utf-8") as f:
            out.append(json.load(f))
    return out
def _ms(vals):
    if len(vals)>1:
        return f"{statistics.mean(vals):+.5f}±{statistics.stdev(vals):.5f}"
    return f"{vals[0]:+.5f}"

def summarize(args):
    nr_runs=_load(os.path.join(RESULTS_DIR,"runs","nodereplay_*","result.json"))
    a2_runs=_load(os.path.join(A2_RESULTS_DIR,"runs","*","result.json"))
    if not nr_runs:
        raise SystemExit("No finished NodeReplay runs found.")
    if not a2_runs:
        raise SystemExit(f"No A2 runs found in {A2_RESULTS_DIR}")

    groups={}
    for r in a2_runs+nr_runs:
        groups.setdefault(_group_key(r),[]).append(r)

    node_cols=[f"N{n} dDomain" for n in A2.NODES]
    header=["method","seeds","dGeneral(forget)"]+node_cols+["mean rho","replay tokens"]
    rows=[]
    for key,rs in sorted(groups.items()):
        fg=[r["forgetting"] for r in rs]
        nodes=[[r[f"node{n}_domain_improvement"]["value"] for r in rs] for n in A2.NODES]
        rho=[r["mean_rho"] if "mean_rho" in r else r["replay_ratio"] for r in rs]
        toks=[r["total_replay_tokens"] for r in rs if r.get("total_replay_tokens") is not None]
        rows.append([
            key,str(len(rs)),_ms(fg),*[_ms(v) for v in nodes],
            f"{statistics.mean(rho):.3f}",
            f"{statistics.mean(toks):.0f}" if toks else "n/a"
        ])

    widths=[max(len(str(x)) for x in col) for col in zip(header,*rows)]
    def line(cells): return " | ".join(str(c).ljust(w) for c,w in zip(cells,widths))
    print(line(header)); print("-+-".join("-"*w for w in widths))
    for row in rows: print(line(row))

    # Paired per-seed deltas: NodeReplay vs each A2 baseline.
    # Positive = NodeReplay better (less forgetting / more domain improvement).
    nr_by_seed={r["seed"]:r for r in nr_runs}
    print("\nNodeReplay minus baseline, paired by seed (positive = NodeReplay better)")
    h2=["baseline","seeds","d(-forget)"]+[f"N{n} d(impr)" for n in A2.NODES]
    rows2=[]
    for key,rs in sorted(groups.items()):
        if not key.startswith("A2_fixed"): continue
        pairs=[(nr_by_seed[r["seed"]],r) for r in rs if r["seed"] in nr_by_seed]
        if not pairs: continue
        df=[b["forgetting"]-n["forgetting"] for n,b in pairs]
        dn=[[n[f"node{k}_domain_improvement"]["value"]-b[f"node{k}_domain_improvement"]["value"]
             for n,b in pairs] for k in A2.NODES]
        rows2.append([key,str(len(pairs)),_ms(df),*[_ms(v) for v in dn]])
    w2=[max(len(str(x)) for x in col) for col in zip(h2,*rows2)]
    print(" | ".join(str(c).ljust(w) for c,w in zip(h2,w2)))
    print("-+-".join("-"*w for w in w2))
    for row in rows2: print(" | ".join(str(c).ljust(w) for c,w in zip(row,w2)))

    # Best fixed ratio in hindsight (final-eval slices; A2 has no validation signals)
    scored={}
    for k,rs in groups.items():
        if k.startswith("A2_fixed"):
            scored[k]=statistics.mean(
                -statistics.mean(r[f"node{n}_domain_improvement"]["value"] for n in A2.NODES)
                +args.lam*r["forgetting"] for r in rs)
    best=min(scored,key=scored.get)
    print(f"\nBest A2 fixed ratio in hindsight (lambda={args.lam}, chosen on eval slices): {best}")
    print("-> report this as an optimistic baseline for a global fixed ratio.")

    out=os.path.join(RESULTS_DIR,"summary.json")
    with open(out,"w",encoding="utf-8") as f:
        json.dump({k:[r.get("run_name",f"A2_seed{r['seed']}") for r in rs] for k,rs in groups.items()},f,indent=2)
    print(f"Saved run index: {out}")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["calibrate","suggest","run","fixed","summarize"],required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--init_rho",type=float,default=0.10)
    parser.add_argument("--node_rhos",type=str,default=None,
                        help="fixed mode only: comma list, one rho per node (per-node oracle baseline)")
    parser.add_argument("--rounds",type=int,default=None)
    parser.add_argument("--local_steps",type=int,default=A2.LOCAL_STEPS)
    parser.add_argument("--tau_f",type=float,default=None)
    parser.add_argument("--tau_low",type=float,default=None)
    parser.add_argument("--tau_s",type=float,default=None)
    parser.add_argument("--lam",type=float,default=1.0,
                        help="summarize: weight of general forgetting when picking best fixed ratio")
    parser.add_argument("--tag",type=str,default="")
    args=parser.parse_args()

    if args.init_rho not in RHO_GRID:
        raise SystemExit(f"--init_rho must be one of {RHO_GRID}")

    if args.rounds is None:
        args.rounds=3 if args.mode=="calibrate" else A2.ROUNDS

    if args.mode=="calibrate":
        calibrate(args)
    elif args.mode=="suggest":
        suggest(args)
    elif args.mode=="run":
        execute(args,adaptive=True)
    elif args.mode=="fixed":
        execute(args,adaptive=False)
    else:
        summarize(args)