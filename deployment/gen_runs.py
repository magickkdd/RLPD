#!/usr/bin/env python
"""全阶段 run 清单生成器（v5：命令构造按脚本能力分流 + 真自检）。
脚本 flag 能力矩阵（实测）:
  train_finetuning.py / train_baselines.py    : --checkpoint_model ✓ --project_name ✓ --log_dir ✓
  train_finetuning_pixels.py / train_baselines_pixels.py: 无 checkpoint_model，用 --save_dir ✓ --project_name ✓
  final_eval.py                               : 仅 env_name/seed/eval_episodes/ckpt_dir/out_path/config——
                                                **无 checkpoint_model、无 project_name**
用法: python gen_runs.py --suite <name> [--out ...] [--dry-run]
"""
import argparse
import subprocess
import sys

PY_ = "/mnt/data/miniforge3/envs/rlpd311/bin/python"
REPO = "/mnt/workspace/rlpd"
CKPT = "/mnt/workspace/ckpt"
LOGS = "/mnt/workspace/logs"
ENV_PRE = (
    f"cd {REPO} && export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/mnt/workspace/mujoco210/bin:$LD_LIBRARY_PATH"
    " && export WANDB_MODE=offline && export XLA_PYTHON_CLIENT_PREALLOCATE=false"
)
SEEDS = [42, 43, 44]

def cmd(script, flags, log, project=None):
    """按脚本能力分流追加 flag（v22 四个 FATAL 的根因是无差别追加）。"""
    parts = [ENV_PRE, f"{PY_} {script} {flags}"]
    if script in ("final_eval.py",):
        pass  # final_eval 无 project_name/checkpoint_model
    elif script in ("train_finetuning_pixels.py", "train_baselines_pixels.py"):
        parts.append(f"--project_name={project}")  # checkpoint 走 --save_dir（已在 flags 里）
    else:
        parts.append(f"--project_name={project} --checkpoint_model=True")
    parts.append(f"> {log} 2>&1")
    return " && ".join(parts[:-1]) + " " + parts[-1] if False else " ".join(
        [" && ".join(parts[:-1]), parts[-1]]
    )

def exp_prefix(seed, ln):
    return f"s{seed}_0pretrain" + ("_LN" if ln else "")

def suite_p2():
    tasks = [("halfcheetah-expert-v0", 200000), ("hopper-expert-v0", 200000),
             ("walker2d-expert-v0", 200000), ("walker2d-medium-replay-v0", 250000)]
    out = []
    for env, steps in tasks:
        for seed in SEEDS:
            base = f"--env_name={env} --seed={seed} --start_training=5000 --max_steps={steps}"
            out.append(cmd("train_finetuning.py",
                           f"{base} --utd_ratio=20 --config=configs/rlpd_config.py "
                           f"--log_dir={CKPT}/p2/RLPD/{env}/s{seed}",
                           f"{LOGS}/p2_RLPD_{env}_s{seed}.log", "rlpd_p2"))
            out.append(cmd("train_baselines.py",
                           f"{base} --utd_ratio=1 --baseline=sacfd --config=configs/sac_config.py "
                           f"--log_dir={CKPT}/p2/SACfD/{env}/s{seed}",
                           f"{LOGS}/p2_SACfD_{env}_s{seed}.log", "rlpd_p2"))
            out.append(cmd("train_baselines.py",
                           f"{base} --utd_ratio=20 --baseline=online --config=configs/rlpd_config.py "
                           f"--log_dir={CKPT}/p2/Online/{env}/s{seed}",
                           f"{LOGS}/p2_Online_{env}_s{seed}.log", "rlpd_p2"))
    return out

def suite_p3():
    amaze_cfg = ('--config=configs/rlpd_config.py --config.backup_entropy=False '
                 '--config.hidden_dims="(256, 256, 256)"')
    out = []
    for seed in SEEDS:
        out.append(cmd("train_finetuning.py",
                       f"--env_name=antmaze-umaze-v2 --utd_ratio=20 --start_training=5000 --max_steps=300000 "
                       f"{amaze_cfg} --config.num_min_qs=1 --seed={seed} "
                       f"--log_dir={CKPT}/p3/RLPD/antmaze-umaze-v2/s{seed}",
                       f"{LOGS}/p3_RLPD_antmaze-umaze-v2_s{seed}.log", "rlpd_p3"))
        for qs, algo in [("1", "RLPD_LD"), ("2", "CDQon_LD")]:
            out.append(cmd("train_finetuning.py",
                           f"--env_name=antmaze-large-diverse-v2 --utd_ratio=20 --start_training=5000 "
                           f"--max_steps=300000 {amaze_cfg} --config.num_min_qs={qs} --seed={seed} "
                           f"--log_dir={CKPT}/p3/{algo}/antmaze-large-diverse-v2/s{seed}",
                           f"{LOGS}/p3_{algo}_antmaze-large-diverse-v2_s{seed}.log", "rlpd_p3"))
    return out

def suite_p4():
    return [cmd("train_finetuning.py",
                "--env_name=pen-binary-v0 --utd_ratio=20 --start_training=0 --max_steps=1000 "
                '--config=configs/rlpd_config.py --config.backup_entropy=False '
                '--config.hidden_dims="(256, 256, 256)" --seed=42 '
                f"--log_dir={CKPT}/p4/RLPD/pen-binary-v0/s42",
                f"{LOGS}/p4_smoke_s42.log", "rlpd_p4")]

def suite_p5():
    out = [cmd("train_finetuning_pixels.py",
               "--env_name=cheetah-run-v0 --action_repeat=2 --start_training=0 --max_steps=10000 "
               "--config=configs/rlpd_pixels_config.py --seed=42 "
               f"--save_dir={CKPT}/p5/calib/cheetah-run/s42/ckpt",
               f"{LOGS}/p5_calib_s42.log", "rlpd_p5")]
    for seed in SEEDS:
        out.append(cmd("train_finetuning_pixels.py",
                       "--env_name=cheetah-run-v0 --action_repeat=2 --start_training=5000 --max_steps=300000 "
                       "--config=configs/rlpd_pixels_config.py "
                       f"--seed={seed} --save_dir={CKPT}/p5/RLPD/cheetah-run/s{seed}/ckpt",
                       f"{LOGS}/p5_RLPD_cheetah-run_s{seed}.log", "rlpd_p5"))
        out.append(cmd("train_baselines_pixels.py",
                       "--baseline=online --env_name=cheetah-run-v0 --action_repeat=2 --start_training=5000 "
                       "--max_steps=300000 --config=configs/rlpd_pixels_config.py --offline_ratio=0 "
                       f"--seed={seed} --save_dir={CKPT}/p5/Online/cheetah-run/s{seed}/ckpt",
                       f"{LOGS}/p5_Online_cheetah-run_s{seed}.log", "rlpd_p5"))
    return out

def suite_p6():
    out = []
    for seed in SEEDS:
        out.append(cmd("train_finetuning.py",
                       "--env_name=walker2d-expert-v0 --utd_ratio=20 --start_training=5000 --max_steps=200000 "
                       "--config=configs/rlpd_config.py --config.critic_layer_norm=False "
                       f"--seed={seed} --log_dir={CKPT}/p6/NoLN/walker2d-expert-v0/s{seed}",
                       f"{LOGS}/p6_NoLN_walker2d-expert-v0_s{seed}.log", "rlpd_p6"))
        out.append(cmd("train_baselines.py",
                       "--baseline=sacfd --config=configs/rlpd_config.py "
                       "--env_name=walker2d-medium-v0 --utd_ratio=20 --start_training=5000 --max_steps=200000 "
                       f"--seed={seed} --log_dir={CKPT}/p6/Seeded/walker2d-medium-v0/s{seed}",
                       f"{LOGS}/p6_Seeded_walker2d-medium-v0_s{seed}.log", "rlpd_p6"))
        out.append(cmd("train_finetuning.py",
                       "--env_name=walker2d-medium-v0 --utd_ratio=20 --start_training=5000 --max_steps=200000 "
                       "--config=configs/rlpd_config.py "
                       f"--seed={seed} --log_dir={CKPT}/p6/Symmetric/walker2d-medium-v0/s{seed}",
                       f"{LOGS}/p6_Symmetric_walker2d-medium-v0_s{seed}.log", "rlpd_p6"))
    return out

def _fe(algo, env, cfg, ln, seed, ckpt_root, project):
    ck = f"{ckpt_root}/{algo}/{env}/s{seed}/{exp_prefix(seed, ln)}/checkpoints"
    return cmd("final_eval.py",
               f"--env_name={env} --config={cfg} --seed={seed} --eval_episodes=100 --ckpt_dir={ck}",
               f"{LOGS}/fe_{ckpt_root.split('/')[-1]}_{algo}_{env}_s{seed}.log")

def suite_p2_final_eval():
    out = []
    for env in ["halfcheetah-expert-v0", "hopper-expert-v0", "walker2d-expert-v0",
                "walker2d-medium-replay-v0"]:
        for algo, cfg, ln in [("RLPD", "configs/rlpd_config.py", True),
                              ("SACfD", "configs/sac_config.py", False),
                              ("Online", "configs/rlpd_config.py", True)]:
            for seed in SEEDS:
                out.append(_fe(algo, env, cfg, ln, seed, f"{CKPT}/p2", "rlpd_p2"))
    return out

def suite_p3_final_eval():
    """antmaze 100 trials 是论文 B.1 明确规定（非可选）。CDQon_LD 的 exp_prefix 含 _LN（rlpd_config LN=True）。"""
    out = []
    for algo in ["RLPD", "RLPD_LD", "CDQon_LD"]:
        env = "antmaze-umaze-v2" if algo == "RLPD" else "antmaze-large-diverse-v2"
        for seed in SEEDS:
            out.append(_fe(algo, env, "configs/rlpd_config.py", True, seed, f"{CKPT}/p3", "rlpd_p3"))
    return out

def suite_p6_final_eval():
    out = []
    for algo in ["NoLN", "Seeded", "Symmetric"]:
        env = "walker2d-expert-v0" if algo == "NoLN" else "walker2d-medium-v0"
        for seed in SEEDS:
            ln = algo != "NoLN"  # NoLN 的 config.critic_layer_norm=False → exp_prefix 无 _LN
            out.append(_fe(algo, env, "configs/rlpd_config.py", ln, seed, f"{CKPT}/p6", "rlpd_p6"))
    return out

SUITES = {
    "p2": (suite_p2, 36), "p3": (suite_p3, 9), "p4": (suite_p4, 1),
    "p5": (suite_p5, 7), "p6": (suite_p6, 9),
    "p2_final_eval": (suite_p2_final_eval, 36),
    "p3_final_eval": (suite_p3_final_eval, 9),
    "p6_final_eval": (suite_p6_final_eval, 9),
}

def selfcheck(cmds, expect, out):
    """真自检：期望行数断言 + bash -n 全行语法检查（v23：行数正确 ≠ 可运行）。"""
    assert len(cmds) == expect, f"行数 {len(cmds)} != 期望 {expect}"
    for i, c in enumerate(cmds):
        r = subprocess.run(["bash", "-n"], input=c, text=True, capture_output=True)
        assert r.returncode == 0, f"第 {i+1} 行 bash 语法错误: {r.stderr[:200]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    fn, expect = SUITES[args.suite]
    cmds = fn()
    selfcheck(cmds, expect, args.suite)  # 每次生成都跑
    text = "\n".join(cmds) + "\n"
    if args.dry_run:
        print(text)
    else:
        out = args.out or f"runs_{args.suite}.txt"
        open(out, "w").write(text)
        print(f"wrote {len(cmds)} lines (selfcheck OK) -> {out}")

if __name__ == "__main__":
    main()
