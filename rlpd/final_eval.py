#! /usr/bin/env python
"""P2 终评（100 episodes；论文 B.1 的 100 trials 仅规定 AntMaze/Adroit，locomotion 系本复现扩展口径）:
加载训练 run 的最新 checkpoint，评估并写 {ckpt_dir 的父目录}/final_eval.json。

用法:
  python final_eval.py --env_name=halfcheetah-expert-v0 \
      --config=configs/rlpd_config.py --seed=42 \
      --ckpt_dir=/mnt/workspace/ckpt/p2/RLPD/halfcheetah-expert-v0/s42/s42_0pretrain_LN/checkpoints
"""
import json
import os
import re

import tensorflow  # TF 先行：防 orbax 二次初始化 segfault（v24 实测，teardown abort 无害）

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
import numpy as np
from absl import app, flags

from flax.training import checkpoints
from ml_collections import config_flags

from rlpd.agents import SACLearner
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v0", "D4RL dataset name.")
flags.DEFINE_integer("seed", 42, "Random seed (agent init / env seed).")
flags.DEFINE_integer("eval_episodes", 100, "Final evaluation episodes.")
flags.DEFINE_string("ckpt_dir", None, "Path to .../checkpoints directory.")
flags.DEFINE_string("out_path", None, "Output json (default: <ckpt_dir>/../final_eval.json).")
flags.DEFINE_integer("ckpt_step", 0, "Restore this checkpoint step (0 = latest).")

config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    assert FLAGS.ckpt_dir is not None

    print("[probe] make_env", flush=True)
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env.seed(FLAGS.seed)
    print("[probe] env_ok", flush=True)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )
    print("[probe] agent_ok", flush=True)

    # restore_checkpoint(target=agent) 按 agent 结构还原为 dataclass（flax 序列化 API；
    # step=None 取目录内最新 step）
    latest = checkpoints.latest_checkpoint(FLAGS.ckpt_dir)
    if latest is None:
        # 硬失败（v22）：训练端 save 被 try/except 静默包裹，若 ckpt 缺失绝不允许输出垃圾分数
        print(f"[final_eval] FATAL: no checkpoint in {FLAGS.ckpt_dir}", flush=True)
        raise SystemExit(1)
    # latest_checkpoint 返回路径字符串（.../checkpoint_0），从尾部提取步数（v23 修正：int(路径) 会 ValueError）
    m = re.search(r"(\d+)\s*/?\s*$", latest.rstrip("/"))
    ckpt_step = int(m.group(1)) if m else None
    actual_step = FLAGS.ckpt_step or ckpt_step
    agent = checkpoints.restore_checkpoint(
        FLAGS.ckpt_dir, target=agent, step=FLAGS.ckpt_step or None
    )
    print("[probe] restore_ok", flush=True)
    print(f"[final_eval] restored ckpt (step={actual_step}) from {FLAGS.ckpt_dir}", flush=True)

    eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
    ret = float(eval_info["return"])
    length = float(eval_info["length"])
    # 归一化：d4rl OfflineEnv 有 get_normalized_score（0~1 → ×100 对齐论文口径）；
    # mj_envs env（pen-binary 等）/像素 env 无此方法 → normalized_score = None（原始 return 兜底）
    if hasattr(env, "get_normalized_score"):
        norm = float(env.get_normalized_score(ret)) * 100.0
    else:
        norm = None

    out = FLAGS.out_path or os.path.join(os.path.dirname(FLAGS.ckpt_dir.rstrip("/")), "final_eval.json")
    with open(out, "w") as f:
        json.dump(
            dict(
                env_name=FLAGS.env_name,
                seed=FLAGS.seed,
                episodes=FLAGS.eval_episodes,
                ckpt_step=actual_step,
                return_raw=ret,
                length=length,
                normalized_score=norm,  # None = 无 REF 口径（Adroit mj_envs/像素），aggregate 用原始 return
                normalized_score_available=norm is not None,
            ),
            f,
            indent=2,
        )
    print(f"[final_eval] return={ret:.2f} normalized={norm:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    app.run(main)
