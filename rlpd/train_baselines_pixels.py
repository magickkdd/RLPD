#! /usr/bin/env python
"""P5 Online 基线（派生自 train_finetuning_pixels.py，仅改数据通路）:
删除离线数据流（ds_iterator 与 combine），只从在线 replay buffer 采样。
必须配合 --offline_ratio=0 使在线迭代器 batch = batch_size*utd*(1-0) = 256，
与 RLPD 组的 batch 对齐（唯一变量 = 有无离线数据）。
注意: 像素脚本无 --checkpoint_model flag，checkpoint 用 --save_dir（save_checkpoint 每 eval_interval 保存）。
"""
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
from rlpd.agents import DrQLearner
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.evaluation import evaluate
from rlpd.wrappers import WANDBVideo, wrap_pixels

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd_pixels_online", "wandb project name.")
flags.DEFINE_enum("baseline", "online", ["online"], "Baseline type (only online).")
flags.DEFINE_string("env_name", "cheetah-run-v0", "dm_control env name.")
flags.DEFINE_string("dataset_level", "medium", "V-D4RL dataset level (unused for online).")
flags.DEFINE_string("dataset_path", None, "Path to dataset (unused for online).")
flags.DEFINE_integer("dataset_size", 500_000, "Unused for online (kept for compat).")
flags.DEFINE_float("offline_ratio", 0.0, "Must be 0 for online baseline.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training env steps.")
flags.DEFINE_integer(
    "start_training", int(5e3), "Number of steps to start training."
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_integer("action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("memory_efficient_replay_buffer", True, "Use a memory efficient replay buffer.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_string("save_dir", None, "Directory to save checkpoints.")
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Replay buffer size; defaults to max_steps // action_repeat."
)

config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    assert FLAGS.offline_ratio == 0.0, "online 基线必须 --offline_ratio=0（batch 对齐 RLPD 组）"

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    action_repeat = FLAGS.action_repeat or 2

    def wrap(env):
        camera_id = 2 if "quadruped" in FLAGS.env_name else 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )

    env = gym.make(FLAGS.env_name)
    env, pixel_keys = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.save_video:
        env = WANDBVideo(env)
    env.seed(FLAGS.seed)
    # online: 无离线数据集（不构建 VD4RLDataset）

    eval_env = gym.make(FLAGS.env_name)
    eval_env, _ = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.max_steps // action_repeat
    if FLAGS.memory_efficient_replay_buffer:
        replay_buffer = MemoryEfficientReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(FLAGS.batch_size * FLAGS.utd_ratio),  # = 256（offline_ratio=0）
                "pack_obs_and_next_obs": True,  # MemoryEfficientReplayBuffer 必须 packed，否则 drq._unpack UnboundLocalError
            }
        )
    else:
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(FLAGS.batch_size * FLAGS.utd_ratio),
            }
        )

    replay_buffer.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        pixel_keys=pixel_keys,
        **kwargs,
    )

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
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
                wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)  # 无 combine，单 buffer
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)

            if FLAGS.save_dir is not None:
                from flax.training import checkpoints

                checkpoints.save_checkpoint(
                    FLAGS.save_dir, target=agent, step=i * action_repeat, overwrite=True
                )


if __name__ == "__main__":
    app.run(main)
