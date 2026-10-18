import h5py, numpy as np, glob, os

# d4rl REF 表（v0 口径）: random / expert
REF = {
    'halfcheetah': (-280.1789532, 12135.0),
    'hopper': (-20.272305, 3234.3),
    'walker2d': (1.629008, 4592.3),
}

for path in sorted(glob.glob('/mnt/workspace/.d4rl/datasets/*_expert.hdf5') + glob.glob('/mnt/workspace/.d4rl/datasets/walker2d_medium.hdf5') + glob.glob('/mnt/workspace/.d4rl/datasets/walker_mixed.hdf5')):
    name = os.path.basename(path).replace('.hdf5', '')
    try:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            rews = f['rewards'][:]
            term = f['terminals'][:]
            timeouts = f['timeouts'][:] if 'timeouts' in keys else None
            n = len(rews)
            ends = np.where(term.astype(bool))[0]
            if timeouts is not None:
                ends = np.union1d(ends, np.where(timeouts.astype(bool))[0])
            starts = np.concatenate([[0], ends[:-1] + 1])
            rets = np.add.reduceat(rews, starts)
            rets = rets[:-1] if len(rets) > 1 else rets
            fam = name.split('_')[0]
            r0, rX = REF.get(fam, (0, 1))
            norm_mean = float(np.mean((rets - r0) / (rX - r0)) * 100)
            print(f"{name}: N={n} keys={keys}")
            print(f"  reward mean={rews.mean():.3f} max={rews.max():.3f} term%={term.mean()*100:.1f}")
            print(f"  n_traj~={len(rets)} traj_return mean={rets.mean():.1f} p50={np.percentile(rets,50):.1f} max={rets.max():.1f} -> norm mean={norm_mean:.1f}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
