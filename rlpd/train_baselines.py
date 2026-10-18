#! /usr/bin/env python
"""P2 对照基线（派生自 train_finetuning.py，仅改数据通路，算法不动）:
  --baseline=sacfd  : 论文 §5 定义的 SACfD = plain SAC（configs/sac_config.py: num_qs=2、无 LN、2 层）
                      + "initializes the online replay buffer with the offline data"
                      （固定容量 = 数据集大小，在线数据按 FIFO 循环覆盖、逐渐稀释）
  --baseline=online : REDQ 结构（configs/rlpd_config.py）但无离线数据（纯在线对照）
注意: offline_ratio=0/1 会在 combine() 触发 broadcast 错误，故基线走独立采样路径而非 flag。
"""
import os
import pickle
import time

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset

try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd_baselines", "wandb project name.")
flags.DEFINE_enum("baseline", "sacfd", ["sacfd", "online"], "Baseline type.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates (sacfd only).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
# 提示: sacfd 应显式传 --config=configs/sac_config.py（plain SAC）；
#       online 应显式传 --config=configs/rlpd_config.py（REDQ 结构）


def main(_):
    assert FLAGS.pretrain_steps == 0, "train_baselines 对照设计 pretrain=0（pretrain 循环已移除）"
    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"
    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    ds = None
    if FLAGS.baseline == "sacfd":
        ds = D4RLDataset(env)
        # 固定容量 = 数据集大小: 在线数据 FIFO 循环覆盖最早离线数据（论文 "initialize" 语义）
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, len(ds)
        )
    else:  # online: 无离线数据
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.max_steps
        )
    replay_buffer.seed(FLAGS.seed)

    if ds is not None:
        # 预灌离线数据（逐条走正常 insert 路径，含 masks/dones 语义），计时入日志
        t0 = time.time()
        n = len(ds)
        for j in range(n):
            replay_buffer.insert(
                dict(
                    observations=ds.dataset_dict["observations"][j],
                    next_observations=ds.dataset_dict["next_observations"][j],
                    actions=ds.dataset_dict["actions"][j],
                    rewards=ds.dataset_dict["rewards"][j],
                    masks=ds.dataset_dict["masks"][j],
                    dones=ds.dataset_dict["dones"][j],
                )
            )
        print(
            f"[seeded] baseline={FLAGS.baseline} inserted {n} transitions "
            f"in {time.time() - t0:.1f}s, buffer_len={len(replay_buffer)}",
            flush=True,
        )

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    resume_step = 0
    if FLAGS.checkpoint_model:
        _latest = checkpoints.latest_checkpoint(chkpt_dir)
        if _latest:
            agent = checkpoints.restore_checkpoint(chkpt_dir, target=agent, step=None)
            resume_step = int(str(_latest).rstrip("/").rsplit("_", 1)[1])
            print(f"[resume] continue from step {resume_step}", flush=True)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(resume_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

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
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)

        if i >= FLAGS.start_training:
            # sacfd/online: 单一 buffer 均匀采样（无对称采样与 combine）
            batch = replay_buffer.sample(int(FLAGS.batch_size * FLAGS.utd_ratio))

            if "antmaze" in FLAGS.env_name:
                batch = dict(batch)
                batch["rewards"] -= 1

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")


if __name__ == "__main__":
    app.run(main)
