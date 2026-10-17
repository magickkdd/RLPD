#!/mnt/data/miniforge3/envs/rlpd311/bin/python
"""pen-cloned-v1.hdf5 (d4rl 格式) -> AWAC rlkit npy 格式（仅用于 G4 冒烟链路验证）。
产出: ~/.datasets/awac-data/{pen2_sparse.npy, pen_bc_sparse4.npy}
注意: 稠密奖励数据, 不可用于结果级（P4_SPEC §4 路径③降级理由）。"""
import h5py, numpy as np, sys

D = '/root/.datasets/awac-data'
f = h5py.File(f'{D}/pen-cloned-v1.hdf5', 'r')
print('hdf5 keys:', list(f.keys()))
obs = f['observations'][:]; act = f['actions'][:]
rew = f['rewards'][:]; term = f['terminals'][:].astype(np.float32)
# v1 数据无 next_observations: 按边界自环派生（episode 末 next=obs[i], 其余 obs[i+1]）
N0 = len(rew)
bounds = (term == 1.0)
for i in range(N0 - 1):
    if (i + 1) % 100 == 0:
        bounds[i] = True  # pen horizon=100 timeout
nobs = np.empty_like(obs)
nobs[:-1] = obs[1:]
nobs[:-1][bounds[:-1]] = obs[:-1][bounds[:-1]]
nobs[-1] = obs[-1]
print('self-loop boundaries:', int(bounds.sum()))
N = len(rew)
print(f'N={N} obs_dim={obs.shape[1:]} act_dim={act.shape[1:]} terminals_sum={term.sum()}')

ends = list(np.where(term == 1.0)[0])
if not ends or ends[-1] != N - 1:
    ends.append(N - 1)

def split():
    s = 0
    for e in ends:
        yield s, e + 1
        s = e + 1

ep = [{'observations': [{'state_observation': obs[i]} for i in range(a, b)],
       'next_observations': [{'state_observation': nobs[i]} for i in range(a, b)],
       'actions': [act[i] for i in range(a, b)],
       'rewards': np.append(rew[a:b], 0.0),
       'terminals': term[a:b]} for a, b in split()]
np.save(f'{D}/pen2_sparse.npy', np.array(ep, dtype=object), allow_pickle=True)

bp = [{'observations': obs[a:b], 'next_observations': nobs[a:b], 'actions': act[a:b],
       'rewards': rew[a:b], 'terminals': term[a:b]} for a, b in split()]
np.save(f'{D}/pen_bc_sparse4.npy', np.array(bp, dtype=object), allow_pickle=True)
print(f'expert_paths={len(ep)} bc_paths={len(bp)}')
print('CONVERT_OK')
