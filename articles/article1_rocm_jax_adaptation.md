# 在 AMD MI300X 上复现 RLPD：JAX 0.3→0.8 + ROCm 7.2.3 完整适配之路

> 本文记录我们基于魔搭社区免费 AMD MI300X 实例（192GB 显存），将 ICML 2023 强化学习论文 RLPD 的官方实现（JAX 0.3.x + NVIDIA 栈）完整迁移到 ROCm 7.2.3 + JAX 0.8.x 环境的全过程：API 断裂清单、依赖替代、数值对拍方法，以及最终在 4 个 D4RL 任务 × 3 seeds 上全部达到论文水平的复现结果。

## 一、为什么要做这件事

RLPD（Efficient Online Reinforcement Learning Fine-Tuning with Offline Data）是离线到在线强化学习微调的代表工作：它证明了**不需要保留离线数据集的副本、只需要对称混合采样**，就能在微调阶段高效利用离线数据。官方实现 pin 在 JAX 0.3.x——一个没有 ROCm 7 wheel 的版本。

我们的约束很简单：**只有魔搭社区的免费 AMD MI300X 实例可用**（感谢魔搭与 AMD 的开发者激励计划），而官方代码是为 NVIDIA + JAX 0.3 写的。于是问题变成：JAX 0.3 → 0.8 的大版本跳跃 + NVIDIA → AMD 的硬件迁移，需要付出多少代价？

答案是：**API 层面 ~10 处适配，算法层面零改动**。下面是完整清单。

## 二、JAX 0.3 → 0.8 的 API 断裂清单

### 2.1 FrozenDict.copy 被移除

REDQ 式 ensemble 子采样需要"从 10 个 Q 网络参数里抽 2 个"：

```python
# 旧版（JAX 0.3 + 旧 flax）
params = params.copy(add_or_replace={"Ensemble_0": ens_params})
# 新版（flax 移除了 FrozenDict.copy）
params = frozen_dict.FrozenDict({**params, "Ensemble_0": ens_params})
```

语义完全等价，但 API 不兼容。这类"语法糖移除"在 flax 大版本里最多。

### 2.2 tensorflow-probability 无法使用

RLPD 的 actor 输出 tanh-normal 分布，原实现依赖 tfp 的 `TransformedDistribution`。JAX 0.8 生态里 tfp 没有兼容版本，我们的替代方案是 **~70 行纯 JAX 自实现**（详见第四节——这里藏着本次复现最大的坑）。

### 2.3 flax serialization 与 orbax 的行为差异

`flax.training.checkpoints` 底层从 flax 自研序列化切到了 orbax。两点实测差异：

1. `restore_checkpoint(dir, target=agent)` 的 `target` 参数必须给——它按 target 的 pytree 结构把存档数组对位还原成 dataclass；不给 target 会拿到裸 dict。
2. TF/JAX 混装会触发 segfault：orbax 初始化时拉起 TF runtime，若 `import tensorflow` 不在**所有 import 之前**，XLA 初始化顺序冲突直接 abort。我们的修复是源文件第一行加 `import tensorflow`——一行代码，12/12 个评估进程从全灭到全过。

### 2.4 XLA 编译缓存与 static_argnames

`jax.jit` 对非数组参数必须声明 `static_argnames`，且每个 static 组合独立编译。RLPD 的 `update(batch, utd_ratio=20)` 里 utd_ratio 控制 Python 层 for 循环展开，必须 static。改 utd_ratio = 重新编译——这在调参实验里要心里有数。

## 三、数值对拍：证明"迁移没改算法"

适配最大的风险不是跑不起来，而是**跑起来了但数值悄悄变了**。我们的对拍策略分三层：

1. **叶级对拍**：checkpoint restore 前后 36 条参数路径逐一对拍，零差异（证明 ckpt 链路无罪）。
2. **pretrain 段 loss 曲线对拍**：同一份数据、同一 seed，迁移前后 loss 曲线 1e-6 级重合。
3. **分布对拍**：自实现 tanh-normal 与 tfp 参考实现对比（正常区/临界区/饱和区 × 函数值/梯度值六维）——这一层救了我们一次，也坑了我们一次（详见另一篇排查实录）。

## 四、自实现 tanh-normal：70 行替换 tfp

数学形式本身不难：

```python
# x = tanh(y), y ~ N(mu, sigma^2)
# log p(x) = log p(y) - Σ log(1 - tanh(y_i)^2)
# 数值稳定形式: log(1-tanh²(y)) = 2*(log2 - y - softplus(-2y))
#   （直接算 1-tanh² 在 |y|>9 时灾难性消去：f32 下 tanh(y)=1.0 → log(0) = -inf）
```

真正的难点在**与 tfp 的行为对齐**——包括它不写在文档里的隐式行为（缓存语义），这部分我们踩了个大坑，完整分析见姊妹篇《强化学习训练"猝死"排查实录》。

## 五、性能实测：MI300X 跑 RL 是什么水平

| 指标 | 实测值 |
|---|---|
| 单进程训练吞吐（UTD=20，含仿真） | ~76 steps/s |
| 200k 步单 run 墙钟 | ~40 分钟 |
| 单进程内存 / 显存 | 9.5GB RSS / 76% VRAM（192GB 卡的零头） |
| 推理-only 速度（终评） | ~1500-1850 steps/s |
| 12 run × 100 episodes 终评总耗时 | 12 分钟 |

两个反直觉的发现：

1. **单进程串行快于 4 路并发**：4 路聚合吞吐 54.5 steps/s < 单进程 76。MI300X 的多进程 GPU 调度开销在 RL 这种"每步小算子+高频同步"负载下非常明显。
2. **多进程同时 XLA 初始化会 segfault**：GPU 命令队列（runlist）过订阅，90 秒错峰都救不了训练进程。我们的最终形态是"一波启动后零新进程"。

## 六、最终结果

4 个 D4RL 任务 × 3 seeds，全部跑满目标步数（expert 200k / medium-replay 250k），终评 100 episodes：

| 任务 | seed 42 | seed 43 | seed 44 | 均值 | 论文参考 |
|---|---|---|---|---|---|
| halfcheetah-expert | 99.44 | 95.88 | 104.08 | **99.80** | ~100 |
| hopper-expert | 92.72 | 113.23 | 91.74 | **99.23** | ~100 |
| walker2d-expert | 130.06 | 126.04 | 131.48 | **129.19** | ~105 |
| walker2d-medium-replay | 120.19 | 120.43 | 126.72 | **122.45** | — |

**12/12 全部 ≥ 90 分，4/4 任务均值达到或超过论文水平。**

## 七、给同类迁移的建议

1. **先做叶级对拍再怀疑算法**——迁移故障的归因空间要靠基础设施验证来收窄。
2. **TF/JAX 混装务必让 TF 先 import**——orbax/TF 的初始化顺序问题只有 segfault 一个症状。
3. **`import` 一个包之前先确认它真的被安装了**：`which sshd || apt-get install` 这类短路判断会被同名的平台组件干扰（我们在这上面浪费了三次开机）。
4. **替代第三方库前，把它的隐式状态语义列成清单**——对拍必须覆盖退化/饱和边界，正常区一致 ≠ 等价。

完整代码、训练产物、12 份终评 JSON 与全部排障记录：
[github.com/magickkdd/RLPD](https://github.com/magickkdd/RLPD)

*环境：魔搭社区免费 MI300X Notebook · ROCm 7.2.3 · JAX 0.8.x · Python 3.11*
