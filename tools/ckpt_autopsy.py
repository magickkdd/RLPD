import sys, os
sys.path.insert(0, '/mnt/workspace/rlpd')
os.chdir('/mnt/workspace/rlpd')
import tensorflow
from absl import flags
import jax, numpy as np
import gym, d4rl, d4rl.gym_mujoco, d4rl.locomotion, dmcgym
from flax.training import checkpoints
from ml_collections.config_flags import config_flags
from rlpd.agents import SACLearner
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'hopper-expert-v0', '')
config_flags.DEFINE_config_file('config', 'configs/rlpd_config.py', '')
flags.DEFINE_integer('seed', 42, '')
FLAGS(['probe'])

env = wrap_gym(gym.make('hopper-expert-v0'), rescale_actions=True)
env.seed(0)
kwargs = dict(FLAGS.config)
model_cls = kwargs.pop('model_cls')

def stats(agent, tag):
    print(f"===== {tag} =====")
    for comp in ['actor', 'critic', 'target_critic', 'temp']:
        try:
            obj = getattr(agent, comp)
            params = obj.params
        except AttributeError:
            continue
        leaves, _ = jax.tree_util.tree_flatten(params)
        n_nan = sum(int(np.isnan(np.asarray(l)).sum()) for l in leaves)
        n_inf = sum(int(np.isinf(np.asarray(l)).sum()) for l in leaves)
        norms = [float(np.linalg.norm(np.asarray(l))) for l in leaves]
        mx = max(float(np.abs(np.asarray(l)).max()) for l in leaves)
        print(f"  {comp:14s} n_leaves={len(leaves)} total_norm={np.sqrt(sum(n*n for n in norms)):.1f} max_abs={mx:.2e} nan={n_nan} inf={n_inf}")
    try:
        print(f"  log_alpha={float(agent.temp.params['log_alpha']) if hasattr(agent.temp.params, 'get') else float(agent.temp.params)}")
    except Exception as e:
        print(f"  log_alpha read err: {e}")

fresh = globals()[model_cls].create(42, env.observation_space, env.action_space, **kwargs)
stats(fresh, 'FRESH init (baseline)')

CK = '/mnt/workspace/ckpt/p2/RLPD/%s/s%d/s%d_0pretrain_LN/checkpoints'
for env_name, seed, step in [('hopper-expert-v0', 42, 150000), ('hopper-expert-v0', 42, 105000),
                              ('hopper-expert-v0', 43, 150000), ('walker2d-expert-v0', 42, 150000),
                              ('walker2d-expert-v0', 44, 150000),
                              ('halfcheetah-expert-v0', 42, 150000)]:
    a = globals()[model_cls].create(42, env.observation_space, env.action_space, **kwargs)
    a = checkpoints.restore_checkpoint(CK % (env_name, seed, seed), target=a, step=step)
    stats(a, f"{env_name}/s{seed}@{step}")
