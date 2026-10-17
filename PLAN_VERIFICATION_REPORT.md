# RLPD 复现计划 · 第三方核查报告

> 核查日期: 2026-10-17
> 核查对象: REPRODUCTION_PLAN.md、deployment/ 交付物、rlpd/ 代码、外部报告 A1–A4/B1–B4/F1–F6 断言
> 核查方式: 本地逐文件比对 + 一手来源交叉验证（GitHub Releases API、PyPI、上游仓库源码）
> 结论标注: ✅ 属实 / ⚠️ 部分属实 / ❓ 无法本地复核

---

## 总判定

**外部报告核心结论成立**: A1–A3（AMD 环境不可构建）、B3（冒烟必失败）、B4（tensorflow-cpu 依据错误）、F4（对拍结论靠删测得到）、F5（验收门空占位）均经一手证据确认属实。仅 4 处细节需修正报告本身，其中一处（d4rl 的 Python 版本上限）反而使"不可构建"结论**更硬**。

---

## 逐条核对

### A1. py3.10 决策与 ROCm wheel 矩阵冲突 —— ✅ 属实（且比报告更严重）

- REPRODUCTION_PLAN.md:150 写"rocm jaxlib wheel 仅 cp310~cp313 → 需 py3.10+"；:156 决策"实例内自建 conda py3.10"。
- GitHub API 实证（rocm-jax-v0.8.2 资产全量）: jaxlib 仅有 `+rocm7-cp311/312/313/314` 与 `+rocm7.2.0-cp311/312/313/314` 共 8 个 wheel，**无 cp310**。"cp310~cp313"两头皆错（实际 cp311~cp314）。
- PyPI 实证: jax-rocm7-plugin 最新版 0.11.0.post1（2026-08-14）requires_python **>=3.12**。
- **报告漏写的更硬证据**: d4rl master setup.py 明确 `python_requires=">=3.7, <3.11"`——py3.11/3.12 下 pip 直接拒绝安装。
- **双向夹击**: rocm jaxlib 0.8.2 要求 ≥3.11，d4rl 要求 <3.11 → **不存在任何单一 Python 版本同时满足**。计划的 py3.10 地基不存在，py3.12 退路同样不通。

### A2. wheel URL 404 —— ✅ 属实

- REPRODUCTION_PLAN.md:161 与 setup_amd.sh:66-67 均写死 `jaxlib-0.8.2-cp310-cp310-manylinux2014_x86_64.whl`。
- GitHub API 资产清单无此文件。真实文件名: `jaxlib-0.8.2+rocm7.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`——缺 `+rocm7.2.0` 本地标签、py tag 错、平台 tag 错。
- setup_amd.sh 的 `step()` 包装器"失败继续"，将引发后续连锁失败。

### A3. plugin/pjrt 不带版本必然解析冲突 —— ✅ 结论属实（版本号小误）

- 未 pin 版本确认于三处: REPRODUCTION_PLAN.md:162、setup_amd.sh:71、requirements-rocm.txt:24-25。
- PyPI release history: jax-rocm7-plugin **无任何 0.8.x 版本**（0.7.1 直接跳 0.9.0），最新 0.11.0.post1 要求 py≥3.12。与 jax 0.8.2 配套的 plugin（0.8.2+rocm7.1.1 / rocm7.2.0）**只存在于 GitHub release**，安装单必须改用 release 直链。
- 修正: PyPI 实际最新为 **0.11.0.post1**，非报告所写"0.11.2"。
- `wheelhouse_legacy_rocm7.2.0.zip` 存在/移除细节: ❓ 公开渠道未能独立复核；但"上游弱化 ROCm 7.2 老路线"有旁证（PyPI 无 0.8.x plugin；v0.8.2 的 rocm7.2.0 变体比 release 本体晚一个月补传）。

### A4. 老栈 pin 与 py≥3.12 互斥、无退路 —— ✅ 属实

- requirements-rocm.txt:28-35 实际 pin: `gym==0.23.1`、`mujoco-py<2.2,>=2.1`、`cython<3`、`setuptools<66`、`numpy<2`。
- 计划唯一退路（:161 注释"若无 cp310 则换 py3.12 venv"）: py3.12 下 d4rl（python_requires<3.11）装不上，mujoco-py/gym0.23 构建亦难——**退路是死的**。

### B2. checkpoint 落 cwd 相对路径 —— ✅ 属实

- train_finetuning.py:91-100: `exp_prefix = f"s{seed}_{pretrain_steps}pretrain"` + `_LN`，`chkpt_dir = os.path.join(log_dir, "checkpoints")`。
- rlpd 树内**无 log_dir 的 DEFINE**，用的是 absl.logging 内置 `--log_dir`（默认空串）→ `os.path.join('', prefix)` 退化为 cwd 相对路径。
- setup_amd.sh 持久盘符号链接清单只有 mujoco210/.d4rl/.vd4rl，**无检查点目录**——与 §8.2"检查点落持久盘"矛盾。
- 补充: exp_prefix 不含 env_name，同 seed 不同 env 并发必然同目录冲突（fleet 未实现，设计层风险）；此段与上游原版逐字一致，属上游行为。

### B3. G0a 验收门空转、冒烟必失败 —— ✅ 属实

- REPRODUCTION_PLAN.md:108 G0a 要求"rlpd 1k 步全通"；smoke_test.py 共 9 项，**无一项导入或运行 train_finetuning.py**（最大仅 item 6 单次 agent.update）。
- setup_amd.sh:106 `python "${DIR}/smoke_test.py"`: DIR 为 deployment 目录，sys.path[0] 即该目录；无 sys.path 处理、无 PYTHONPATH、rlpd 未安装 → item 4（rlpd.distributions）与 item 6（rlpd.agents）静态可证必然 ModuleNotFoundError。
- **报告漏写的同类问题**: item 8（import modelscope）必失败（无步骤安装 modelscope）；train_finetuning.py:8 顶层 `import dmcgym` 而 requirements-rocm.txt 未装——G0a 的 1k 步即使修好 PYTHONPATH 也会在 import 阶段挂掉。

### B4. tensorflow-cpu 依据错误、两份安装单不一致 —— ✅ 大体属实（gym 版本号修正）

- requirements-rocm.txt:39-41 注释与安装确认存在。
- d4rl master setup.py `install_requires = gym<0.24.0, numpy, mujoco_py, pybullet, h5py, termcolor, click, dm_control>=1.0.3, mjrl@git`——**确无 tensorflow/tfds**（走 h5py），"白装数百 MB"成立。
- 修正: 报告写"gym=1.0.3"不准确，实际为 `gym<0.24.0`，与计划的 gym==0.23.1 兼容（无害）。
- 两份权威不一致: 确认。REPRODUCTION_PLAN.md:163 正文 `flax>=0.10` 无上界，requirements-rocm.txt:16 为 `flax>=0.10,<0.12`。
- 后果修正: flax 最新 0.12.10 requires-python **≥3.12** → py3.10 环境 pip 只会回退到 0.11.x，**不会**拉到 0.12+；"拉到 0.12+ 并与 jax 0.8 冲突"仅在 py3.12 退路下成立。
- "漏 dmcgym、没有 modelscope": 均属实（dmcgym 被 train_finetuning.py / vd4rl_datasets.py 顶层 import，原版 requirements.txt:14 有而 requirements-rocm.txt 无）。

### F3. 计数错误 —— ✅ 文内矛盾属实；diff 数字 ❓

- REPRODUCTION_PLAN.md:169 "共 4 文件 +67/-29"下列出 1~5 共 5 条（第 5 条为 deployment 包，非代码文件）——确认。
- :174 写"smoke_test.py（8 项）"，实际 9 项，同节后文又写"冒烟项 9"——自相矛盾确认。
- `git diff --stat main rocm-adapt`: ❓ 本地无 .git 元数据（仅工作副本 + rlpd-rocm-bundle.tar.gz），无法复算。
- 顺带: 计划"0.8.2 ↔ ROCm 7.2.1"矩阵表述不精确，release 实际提供 rocm7.1.1 与 rocm7.2.0 两种变体。

### F4. 对拍证据被高估 —— ✅ 属实（一处出处修正）

- **出处修正**: "曾有 6 个用例"出自 .pytest_cache/v/cache/nodeids（6 个测试 id），非 lastfailed；lastfailed 现存 2 条: test_sample_moments、test_sample_identical，**均 FAILED**。当前 test_tanh_normal.py 只剩 4 个测试，被删的正是这两个失败用例。
- 结论链成立: **"4/4 通过"字面为真，"采样统计等价"是靠删除失败测试得到的**。
- test_sampler_moments: 被测侧为手写 `s_new = jnp.tanh(loc + scale * eps)`，**未调用 new.sample()**——batch 采样路径未被对拍。
- test_log_prob_consistent: 容差 `rtol=1e-3, atol=5e-2`，docstring 自述"max abs diff ≈ 1e-6"且自认尾部行为与 tfp 不同（tfp 给 -inf/NaN，自实现 clip 在 1e-6）——与计划宣称"数值一致"非同一判据。
- tanh_transformed.py:50 确认 `clip(value, -1+1e-6, 1-1e-6)`；tanh_normal.py 确认 `log_std_max=2`（std 上限 e²≈7.39）。饱和点 y≈7.25 vs 8.06 与报告"7.26/8.1"吻合。

### F5. 验收门空占位 —— ✅ 属实

§5 表中 G1/G3/G5/G6/G7 验收门列为空，G2 仅"vs 附录 A"（无数值容差/聚合方式/seed 数），G4 仅"依赖链通过"——与"自称按验收门推进"矛盾。

### F6. PRNGKey"legacy API"措辞 —— ⚠️ 部分属实（表述过强）

- REPRODUCTION_PLAN.md:183 确有此句。
- JAX 官方文档在新旧 key 迁移语境中确实称 PRNGKey 产生的 key 为 legacy-style key（新 API 为 jax.random.key/typed key）——"legacy"一词并非无中生有。
- 正确部分: PRNGKey 未弃用、jax 0.8.x 完全可用，计划结论（"保留，预期兼容"）方向没错。属措辞之争，非实质错误。

---

## 报告本身需修正的 4 处

1. **F4 出处**: "lastfailed 显示曾有 6 个用例" → 应为 nodeids 有 6 个，lastfailed 现存 2 条 FAILED。
2. **B4**: d4rl 的 gym 依赖是 `gym<0.24.0` 而非"gym=1.0.3"；且漏写最关键的 `python_requires=">=3.7, <3.11"` 禁止项。
3. **A3**: PyPI plugin 最新为 0.11.0.post1，非 0.11.2。
4. **B4 flax 后果**: "拉到 0.12+"仅在 py3.12 环境成立，py3.10 下 pip 回退到 0.11.x。

## 本地无法复核的 2 项

- F3 的 `git diff --stat` 数字（4 files/+67/-29）——工作区无 .git。
- A3 中 `wheelhouse_legacy_rocm7.2.0.zip` 存在/移除细节——公开渠道未检索到。

---

## "5 个先决问题"前提核实

| # | 问题 | 前提是否成立 |
|---|---|---|
| 1 | py 版本夹击 | ✅ 属实且更硬（≥3.11 vs <3.11 双向夹击）；"分进程"也需跨版本数据交接设计（或改 d4rl 元数据、或绕开 d4rl 直读 h5py） |
| 2 | ~100 runs 清单 | ✅ 计划 §4 仅总数与 ~500 GPU 时，无 config 清单与逐项 run·h |
| 3 | SACfD 配置 | ✅ §3 仅"~30 行 seeded buffer 变体"，无 num_qs/LN/层数/UTD 数值；100-trial 终评在 §8.2 有一句但未落为阶段工作项 |
| 4 | 额度承诺 | ⚠️ 部分属实: :102 确写"现有 100h 即可覆盖主结果"，但同句括注"P1 必须实测并发扩展曲线确认倍率"——是带条件预估而非无条件承诺 |
| 5 | 境内网络与凭据 | ✅ setup_amd.sh 全部直连 github.com/mujoco.org/tailscale.com，无镜像/代理配置；wandb 定 online（D3）无 offline 预案。**另发现**: ms_upload.py 会把含 .secrets.env 的 bundle 上传至 ModelScope 私有仓库——"凭据排除规则"不存在，属真实安全设计缺口 |

---

## 一手来源

- ROCm/rocm-jax release rocm-jax-v0.8.2（GitHub API assets）: https://github.com/ROCm/rocm-jax/releases/tag/rocm-jax-v0.8.2
- jax-rocm7-plugin · PyPI: https://pypi.org/project/jax-rocm7-plugin/
- flax · PyPI: https://pypi.org/project/flax/
- Farama-Foundation/D4RL setup.py（master）: https://github.com/Farama-Foundation/D4RL/blob/master/setup.py
- flax v0.11.0 frozen_dict.py: https://github.com/google/flax/blob/v0.11.0/flax/core/frozen_dict.py
- ikostrikov/rlpd train_finetuning.py（main）: https://github.com/ikostrikov/rlpd/blob/main/train_finetuning.py
- absl.logging 文档（--log_dir 内置 flag）: https://absl.readthedocs.io/en/latest/absl.logging.html
