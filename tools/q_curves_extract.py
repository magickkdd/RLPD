"""从 wandb 0.30 离线文件锚点提取 training/q、temperature、entropy、actor_loss 曲线。
锚点扫描法与 deployment/aggregate.py curves_check 同源（v24 实测有效）。"""
import glob, json, os, re

KEYS = {
    'q': b'\x12\x0atraining/q\x82\x01',
    'temperature': b'\x12\x14training/temperature\x82\x01',
    'entropy': b'\x12\x10training/entropy\x82\x01',
    'actor_loss': b'\x12\x13training/actor_loss\x82\x01',
    'critic_loss': b'\x12\x13training/critic_loss\x82\x01',
}
STP = b'\x12\x05_step\x82\x01'

def scan(raw, kp):
    items = []
    for m in re.finditer(re.escape(kp) + rb'([\x01-\x7f])', raw):
        ln = m.group(1)[0]
        v = raw[m.end():m.end() + ln]
        try:
            items.append((m.start(), float(v.decode())))
        except Exception:
            continue
    return items

def pair(raw, kp):
    """(step, value) 配对：value 与其前方最近的 _step 配对。"""
    vals = scan(raw, kp)
    steps = scan(raw, STP)
    out, si = [], 0
    for pos, v in vals:
        cur = None
        while si < len(steps) and steps[si][0] < pos:
            cur = steps[si][1]
            si += 1
        out.append((cur, v))
    return out

out = {}
for path in sorted(glob.glob('/mnt/workspace/rlpd/wandb/offline-run-*/run-*.wandb')):
    raw = open(path, 'rb').read()
    m_env = re.search(rb"--env_name=([\x21-\x7e/._-]+)", raw)
    m_seed = re.search(rb"--seed=(\d+)", raw)
    if not m_env:
        continue
    env = re.sub(r"[^a-z0-9-]+$", "", m_env.group(1).decode())
    seed = int(m_seed.group(1)) if m_seed else -1
    key = f"{env}/s{seed}"
    run = {}
    for name, kp in KEYS.items():
        pts = pair(raw, kp)
        if pts:
            run[name] = [(int(s) if s is not None else -1, round(v, 4)) for s, v in pts]
    if 'q' in run:
        out[key] = run
        print(key, 'q_pts=', len(run['q']), flush=True)

json.dump(out, open('/mnt/workspace/q_curves.json', 'w'))
print('DONE', len(out), 'runs', flush=True)
