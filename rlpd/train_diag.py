#! /usr/bin/env python
"""v25 诊断实验：对齐正式 run 参数，30k 步内复现猝死 + target_q 插桩 + 可选 q_clip。
与 train_finetuning.py 的差异：无 ckpt/buffer 保存，log/eval 加密，新增 --q_clip。"""
import os

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "diag", "wandb project name.")
flags.DEFINE_string("env_name", "hopper-expert-v0", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Eval episodes.")
flags.DEFINE_integer("log_interval", 500, "Logging interval.")
flags.DEFINE_integer("eval_interval", 2500, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 30001, "Number of training steps.")
flags.DEFINE_integer("start_training", 5000, "Steps to start training.")
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_float("q_clip", 0.0, "Clip target_q to +-q_clip (0 = off).")

config_flags.DEFINE_config_file(
    "config", "configs/rlpd_config.py", "Config.", lock_config=False
)


def combine(one_dict, other_dict):
    combined = {}
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[: v.shape[0]] = v
            tmp[v.shape[0] :] = other_dict[k]
            combined[k] = tmp
    return combined


def main(_):
    wandb.init(project=FLAGS.project_name, name=f"diag_{FLAGS.env_name}_s{FLAGS.seed}_clip{FLAGS.q_clip}")
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env.seed(FLAGS.seed)
    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    ds = D4RLDataset(env)
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    for i in tqdm.tqdm(range(0, FLAGS.pretrain_steps), smoothing=0.1):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        agent, update_info = agent.update(batch=offline_batch, utd_ratio=FLAGS.utd_ratio, q_clip=FLAGS.q_clip)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(0, FLAGS.max_steps + 1), smoothing=0.1):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        mask = 1.0 if (not done or "TimeLimit.truncated" in info) else 0.0
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation
        if done:
            observation, done = env.reset(), False
            for k, v in info.get("episode", {}).items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = ds.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            )
            batch = combine(offline_batch, online_batch)
            agent, update_info = agent.update(batch=batch, utd_ratio=FLAGS.utd_ratio, q_clip=FLAGS.q_clip)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

    print("DIAG_DONE", FLAGS.env_name, FLAGS.seed, FLAGS.q_clip, flush=True)


if __name__ == "__main__":
    app.run(main)
