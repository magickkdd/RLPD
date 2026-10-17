#!/usr/bin/env python
"""P2 V5: 数据可比性 sanity——各数据集 best-episode return 过 get_normalized_score。
expert 数据 best_norm 应 ≥ 95（REF 即数据集最强 episode，正常落 97~100）；结果写入 P2_BENCHMARK.md。
用法: 在实例上 python data_sanity.py [--out P2_BENCHMARK.md]
"""
import argparse
import json
import os

import d4rl  # noqa
import d4rl.gym_mujoco  # noqa
import d4rl.locomotion  # noqa
import gym
import numpy as np

PAIRS = [
    ("halfcheetah-expert-v0", "halfcheetah_expert.hdf5"),
    ("hopper-expert-v0", "hopper_expert.hdf5"),
    ("walker2d-expert-v0", "walker2d_expert.hdf5"),
    ("walker2d-medium-replay-v0", "walker_mixed.hdf5"),
    ("walker2d-medium-v0", "walker2d_medium.hdf5"),
    ("antmaze-umaze-v2", "Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5"),
    ("antmaze-large-diverse-v2", "Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5"),
]


def best_episode_return(rewards, terminals, timeouts):
    ends = np.logical_or(terminals == 1.0, timeouts == 1.0)
    idx = np.where(ends)[0]
    starts = np.concatenate([[0], idx[:-1] + 1])
    return float(np.max([
        rewards[s : e + 1].sum() for s, e in zip(starts, idx)
    ]))


REF_LINES = {"hopper-expert-v0": 100.0, "walker2d-expert-v0": 105.0,
             "halfcheetah-expert-v0": 100.0}  # Fig.19 读图（±5，参考）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench_json = {}
    lines = ["# P2_BENCHMARK（数据 sanity + 论文参考线）", "",
             "| 任务 | 数据文件 | transitions | best episode return | best norm (v0 REF) | 备注 |",
             "|---|---|---|---|---|---|"]
    for env_name, fname in PAIRS:
        path = os.path.expanduser(f"~/.d4rl/datasets/{fname}")
        if not os.path.exists(path):
            lines.append(f"| {env_name} | {fname} | - | - | - | 文件缺失，跳过 |")
            print(f"[SKIP] {fname} 不存在")
            continue
        env = gym.make(env_name)
        ds = env.get_dataset(h5path=path)  # h5path 直读，绕开下载逻辑
        rew = ds["rewards"]
        term = ds.get("terminals", np.zeros_like(rew))
        tout = ds.get("timeouts", np.zeros_like(rew))
        best = best_episode_return(rew, term, tout)
        norm = float(env.get_normalized_score(best)) * 100.0
        note = "OK" if (norm >= 95 if "expert" in env_name else True) else "expert 数据 best_norm<95（REF 即数据集最强 episode，正常应 97~100）——需关注数据强度"
        lines.append(f"| {env_name} | {fname} | {rew.shape[0]} | {best:.1f} | {norm:.1f} | {note} |")
        bench_json[env_name] = dict(file=fname, transitions=int(rew.shape[0]),
                                    best_return=best, best_norm=norm,
                                    fig19_read_ref=REF_LINES.get(env_name))
        print(f"[OK] {env_name}: n={rew.shape[0]} best={best:.1f} best_norm={norm:.1f}")

    lines += ["", "## Fig.19 读图参考线（人工读图，±5 误差，仅参考不作判定；已写入 JSON 供 aggregate 引用）",
              "- hopper-expert-v0 RLPD 终值 ≈ 100；walker2d-expert-v0 ≈ 105；halfcheetah-expert-v0 ≈ 100",
              "- REDQ(Online) 在三任务上均有竞争力（论文 A.3：locomotion 纯在线可较快解决）"]
    print("\n".join(lines))
    out_md = args.out or "/mnt/workspace/P2_BENCHMARK.md"
    open(out_md, "w").write("\n".join(lines) + "\n")
    json_path = (args.out or "/mnt/workspace/P2_BENCHMARK.md").rsplit(".", 1)[0] + ".json"
    json.dump(bench_json, open(json_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out_md} + {json_path}")


if __name__ == "__main__":
    main()
