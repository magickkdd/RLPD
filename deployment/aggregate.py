#!/usr/bin/env python
"""P2+ 聚合对账（v3：按第三轮核查修正归一化/判据/BENCH/glob）。
主判定源 = final_eval.json（100 eps 终评，normalized_score 已 ×100 归一到 0~100 论文口径）；
--curves 时扫 .wandb 做完成度与形状检查（判据 2 用 REF 表把原始 return 归一化，**非原始值比较**）。
用法: python aggregate.py [--root /mnt/workspace/ckpt/p2] [--wandb-dir /mnt/workspace/rlpd/wandb] [--curves] [--out P2_RESULTS.md]
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

DEFAULT_TASKS = ["halfcheetah-expert-v0", "hopper-expert-v0",
                 "walker2d-expert-v0", "walker2d-medium-replay-v0"]
DEFAULT_ALGOS = ["RLPD", "SACfD", "Online"]

# d4rl 1.1 REF 分数（gym_envs.py L23-30，get_normalized_score 返回 0~1 → ×100 对齐论文 0~100 口径）
REF = {
    "halfcheetah-expert-v0": (-280.178953, 12135.0),
    "hopper-expert-v0": (-20.272305, 3234.3),
    "walker2d-expert-v0": (1.629008, 4592.3),
    "walker2d-medium-replay-v0": (1.629008, 4592.3),  # walker2d 系共用 random/expert REF
    "walker2d-medium-v0": (1.629008, 4592.3),          # P6 消融域
    "antmaze-umaze-v2": (0.0, 1.0),                     # P3：成功率直通（0~1 → ×100）
    "antmaze-large-diverse-v2": (0.0, 1.0),             # P3 CDQ 消融域
}


def load_benchmark(path="/mnt/workspace/P2_BENCHMARK.json"):
    """读 data_sanity.py 产出的 JSON（--out 同目录派生 .json）。找不到 → 明确警告（v23：不再静默）。"""
    if not os.path.exists(path):
        print(f"[warn] 基准文件不存在: {path}（data_sanity.py 的 JSON 输出；参考线列将为 '-'）")
        return {}
    try:
        d = json.load(open(path))
        return {k: v.get("fig19_read_ref") for k, v in d.items() if v.get("fig19_read_ref")}
    except Exception as e:
        print(f"[warn] 基准文件解析失败 {path}: {e}")
        return {}


def norm(env, raw):
    lo, hi = REF[env]
    return (raw - lo) / (hi - lo) * 100.0


def load_final_evals(root):
    rows = []
    # ckpt 布局: p2/{algo}/{env}/s{seed}/{exp_prefix}/final_eval.json（final_eval.py 写在 ckpt_dir 父目录）
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "s*", "*", "final_eval.json"))
                    + glob.glob(os.path.join(root, "*", "*", "s*", "final_eval.json"))):
        rel = os.path.relpath(p, root).split(os.sep)
        algo, env, sdir = rel[-4], rel[-3], rel[-2]
        try:
            d = json.load(open(p))
            rows.append(dict(algo=algo, env=env, seed=int(sdir[1:]), **d))
        except Exception as e:
            rows.append(dict(algo=algo, env=env, seed=int(sdir[1:]), error=str(e)))
    return rows


def table(rows, bench, tasks=None, algos=None):
    agg = defaultdict(list)
    for r in rows:
        if r.get("normalized_score") is not None:
            agg[(r["algo"], r["env"])].append(r["normalized_score"])
        elif r.get("return_raw") is not None:
            # 无 REF 口径的 env（Adroit mj_envs/像素）：表内用原始 return 并标记
            agg[(r["algo"], r["env"])].append(f"raw:{r['return_raw']}")
    tasks = tasks or DEFAULT_TASKS
    algos = algos or DEFAULT_ALGOS
    out = ["| 算法 | 任务 | mean±std（0~100） | 各 seed | 论文参考线 |",
           "|---|---|---|---|---|"]
    for algo in algos:
        for env in tasks:
            v = agg.get((algo, env), [])
            ref = bench.get(env, "-")
            if v:
                numeric = [x for x in v if not str(x).startswith("raw:")]
                if numeric:
                    out.append(f"| {algo} | {env} | {np.mean(numeric):.1f} ± {np.std(numeric):.1f} | "
                               f"{', '.join(f'{x:.1f}' for x in sorted(numeric))} | {ref} |")
                else:
                    raws = ", ".join(str(x).replace("raw:", "") for x in sorted(v))
                    out.append(f"| {algo} | {env} | 原始 return: {raws}（无 REF 口径） | - | {ref} |")
            else:
                out.append(f"| {algo} | {env} | （缺） | | {ref} |")
    return "\n".join(out), agg


def g2_judge(agg, curves_summary=None, tasks=None):
    lines = ["### 判据 1（绝对线）", ""]
    v = agg.get(("RLPD", "halfcheetah-expert-v0"), [])
    if v:
        ok1 = len(v) >= 3 and float(np.mean(v)) >= 90.0
        lines.append(f"- halfcheetah-expert-v0 终值归一化分（0~100 口径）≥ 90（出处：论文 Fig.19 读图 ~100 − 10 容差，"
                     f"与 data_sanity best_norm 交叉锚定）：{'PASS' if ok1 else 'FAIL'}"
                     f"（均值 {np.mean(v):.1f}，n={len(v)}）")
    else:
        lines.append("- 终评数据缺失")
    lines += ["", "### 判据 2（样本效率形状，--curves）", ""]
    if curves_summary:
        for env, p50 in curves_summary:
            if "expert" in env and p50 is not None:
                lines.append(f"- {env}: 50k 步内归一化峰值 {p50:.1f} → {'PASS' if p50 >= 60 else 'WATCH'}（≥60）")
            elif p50 is not None:
                lines.append(f"- {env}: 50k 步内归一化峰值 {p50:.1f}（参考）")
        if any("walker2d-medium-v0" in e for e, _ in curves_summary):
            lines.append("- P6 Seeded vs Symmetric：见消融对账表（steps-to-60 + std 描述性呈现，n=3 不作显著性检验）")
    else:
        lines.append("- （未运行 --curves；判据 2 需归一化曲线数据）")
    lines += ["", "### 判据 3（对照，如实报告制）", ""]
    for other in ["SACfD", "Online"]:
        for env in (tasks or DEFAULT_TASKS):
            a = agg.get(("RLPD", env), [])
            b = agg.get((other, env), [])
            if a and b:
                d = float(np.mean(a)) - float(np.mean(b))
                lines.append(f"- RLPD−{other} @ {env}: {d:+.1f} 分（RLPD {np.mean(a):.1f} vs {other} {np.mean(b):.1f}）")
    return "\n".join(lines)


def _split_records(raw: bytes):
    """wandb .wandb 记录帧 = varint 长度前缀 + 内容（v24 实测修正：原固定 4 字节长度假设错误，
    会把记录切碎导致 _step 与指标分家、配对为空）。"""
    off, recs, n = 0, [], len(raw)
    while off < n:
        ln, shift = 0, 0
        while off < n:
            b = raw[off]
            off += 1
            ln |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        if ln <= 0 or off + ln > n:
            break
        recs.append(raw[off : off + ln])
        off += ln
    return recs


def curves_check(wandb_dir):
    """(step, evaluation/return 原始值) 配对 → 用 REF 表归一化（0~100）后做判据 2。
    v24: wandb 0.30 离线文件改用锚点字节扫描（datastore API 已随 0.30 移除）。"""
    RET = b'\x12\x11evaluation/return\x82\x01'
    STP = b'\x12\x05_step\x82\x01'

    def scan(raw):
        items = []
        for kp in (RET, STP):
            for m in re.finditer(re.escape(kp) + rb'([\x01-\x7f])', raw):
                ln = m.group(1)[0]
                v = raw[m.end():m.end() + ln]
                try:
                    items.append((m.start(), kp, float(v.decode())))
                except Exception:
                    continue
        return sorted(items, key=lambda t: t[0])

    def pair(raw):
        rets, cur = [], None
        for _, kp, v in scan(raw):
            if kp is STP:
                cur = v
            else:
                rets.append((cur, v))
        return rets

    rows, summary = [], []
    for path in sorted(glob.glob(os.path.join(wandb_dir, "offline-run-*", "run-*.wandb"))):
        raw = open(path, "rb").read()
        m_env = re.search(rb"--env_name=([\x21-\x7e/._-]+)", raw)
        m_dir = re.search(rb"--log_dir=([\x21-\x7e/._-]+)", raw)
        m_steps = re.search(rb"--max_steps=(\d+)", raw)
        if not (m_env and m_dir):
            continue
        env = re.sub(r"[^a-z0-9-]+$", "", m_env.group(1).decode())
        if env not in REF:
            print(f"[warn] {env} 无 REF 表，跳过曲线归一化（可在 REF dict 补充）")
            continue
        max_steps = int(m_steps.group(1)) if m_steps else 200000
        expect = max_steps // 5000 + 1
        pairs = pair(raw)
        if pairs and all(p[0] is None for p in pairs):
            pairs = [[i * 5000, v] for i, (_, v) in enumerate(pairs)]
        pts = [(int(s), norm(env, v)) for s, v in pairs if s is not None]
        peak50 = max([v for s, v in pts if s <= 50000], default=None)
        step60 = next((st for st, v in pts if v >= 60.0), None)
        rows.append((m_dir.group(1).decode(), env, len(pts), expect, peak50, step60))
        summary.append((env, peak50, step60))
    out = ["| run | eval 条数/期望 | 50k 步内归一化峰值 | 到 60 分步数 | 判据 |", "|---|---|---|---|---|"]
    for ld, env, n, expect, p50, s60 in rows:
        flag = ""
        if p50 is not None and "expert" in env:
            flag = "PASS(50k≥60)" if p50 >= 60 else "WATCH"
        if s60 is not None:
            flag += f" steps_to_60={s60}"
        out.append(f"| {ld.split('/ckpt/')[-1] if '/ckpt/' in ld else ld} | {env} | {n}/{expect} | "
                   f"{p50 and round(p50, 1)} | {s60 or '-'} | {flag} |")
    if not rows:
        out.append("（无 run 数据）")
    return "\n".join(out), summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="p2", help="p2/p3/p5/p6（决定标题/分母/任务算法清单）")
    ap.add_argument("--root", default=None)
    ap.add_argument("--wandb-dir", default="/mnt/workspace/rlpd/wandb")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = {
        "p2": dict(root="/mnt/workspace/ckpt/p2", title="P2 结果对账", total=36,
                   tasks=DEFAULT_TASKS, algos=DEFAULT_ALGOS, out="P2_RESULTS.md"),
        "p3": dict(root="/mnt/workspace/ckpt/p3", title="P3 结果对账", total=9,
                   tasks=["antmaze-umaze-v2", "antmaze-large-diverse-v2"],
                   algos=["RLPD", "RLPD_LD", "CDQon_LD"], out="P3_RESULTS.md"),
        "p5": dict(root="/mnt/workspace/ckpt/p5", title="P5 结果对账", total=6,
                   tasks=["cheetah-run-v0"], algos=["RLPD", "Online"], out="P5_RESULTS.md"),
        "p6": dict(root="/mnt/workspace/ckpt/p6", title="P6 消融对账", total=9,
                   tasks=["walker2d-expert-v0", "walker2d-medium-v0"],
                   algos=["NoLN", "Seeded", "Symmetric"], out="P6_RESULTS.md"),
    }[args.suite]
    root = args.root or cfg["root"]

    bench = load_benchmark()
    rows = load_final_evals(root)
    tab, agg = table(rows, bench, cfg["tasks"], cfg["algos"])
    done = sum(1 for r in rows if r.get("normalized_score") is not None)
    ctab = "（未运行 --curves）"
    curves_summary = None
    if args.curves:
        ctab, curves_summary = curves_check(args.wandb_dir)
    else:
        ctab = "（未运行 --curves）"
    judgment = g2_judge(agg, curves_summary, cfg["tasks"])  # 单段判定（v23）
    body = [
        f"# {cfg['title']}（aggregate v5 自动生成）", "",
        f"- 终评完成: **{done}/{cfg['total']}** runs（100 eps；normalized_score = 0~100 口径，d4rl 0~1 已 ×100；None = 无 REF 的 env 用原始 return）",
        f"- 数据口径: hf-mirror v2 数据喂 -v0 env（P2_BENCHMARK.json 有 best-in-dataset sanity）", "",
        "## 对账表", "", tab, "",
        "## 曲线完成度与形状", "", ctab, "",
        "## 判定（单段）", "", judgment, "",
    ]
    out = args.out or cfg["out"]
    open(out, "w").write("\n".join(body))
    print(f"wrote {out}（终评 {done}/{cfg['total']}）")


if __name__ == "__main__":
    main()
