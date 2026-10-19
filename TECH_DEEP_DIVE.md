# TECH_DEEP_DIVE — 技术细节深潜（面试向问答）

> 全部内容来自本项目的真实代码与实测数据（rlpd/agents/sac/sac_learner.py、distributions/tanh_transformed.py、train_finetuning.py 均逐行读过）。

## 1. RLPD 算法核心机制（会被追问到公式级）

### Q: UTD=20 具体怎么实现的？一次 update 里发生了什么？
```python
@partial(jax.jit, static_argnames=("utd_ratio", "q_clip"))
def update(self, batch, utd_ratio, q_clip=0.0):
    new_agent = self
    for i in range(utd_ratio):
        def slice(x):                    # 对 DatasetDict 每个叶子（obs/act/rew/mask...）切 1/20
            bs = x.shape[0] // utd_ratio
            return x[bs*i : bs*(i+1)]
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, _ = new_agent.update_critic(mini_batch, q_clip)
    new_agent, _ = new_agent.update_actor(mini_batch)      # actor 只更新 1 次，用最后一份
    new_agent, _ = new_agent.update_temperature(actor_info["entropy"])
```
要点：采样端一次采 `batch_size × utd_ratio`（256×20=5120 条），**critic 20 次梯度更新、actor 和温度各 1 次**——这是 RLPD "UTD 只加在 critic 上"的关键设计（actor 高频更新会不稳）。`jax.tree_util.tree_map` 对字典每个数组叶子做同样的切片，保证 mini-batch 对齐。

### Q: offline 和 online 数据怎么混合（对称采样）？
```python
online_batch  = replay_buffer.sample(int(BS*UTD*(1-0.5)))   # 2560 条在线
offline_batch = ds.sample(int(BS*UTD*0.5))                   # 2560 条离线
batch = combine(offline_batch, online_batch)                 # concat
```
`offline_ratio=0.5`：每个 batch 一半来自离线数据集、一半来自在线 replay buffer，50/50 而非按经验加权——这是 RLPD 名字里"对称"的含义。前 5000 步（start_training）只做随机动作填充在线 buffer，之后每步在线采样+训练。

### Q: critic 的 target 怎么算？为什么 ensemble 取 min？
```python
next_qs = target_critic(next_obs, next_actions)   # 10 个 Q
next_q  = next_qs.min(axis=0)                      # REDQ 式子集取 min
target_q = r + γ·mask·next_q − γ·mask·α·log π(next_a|next_s)   # backup_entropy=True
```
- **10 个 Q（num_qs=10）**，target 端**随机抽 2 个（num_min_qs=2）取 min**（`subsample_ensemble`：`jax.random.choice` 不放回抽 idx，再 tree_map 索引参数）——min 抵制过估计，随机子集增加 Q 网络间的独立性。
- **backup_entropy**：target 里减去 γ·mask·α·next_log π，把下一状态的熵也折进回报（soft Q-learning 的完整形式）。
- **actor 更新时用 qs.mean(axis=0) 而不是 min**——policy extraction 用平均 Q 降低方差，target 用 min 抑制过估计，两者分工。

### Q: LayerNorm 放在哪？为什么它对高 UTD 至关重要？
MLP 每个隐藏层的 **pre-activation 位置**（Dense → [Dropout] → **LayerNorm** → ReLU）：
```python
x = nn.Dense(...)(x)
if i+1 < len(hidden_dims) or activate_final:
    x = nn.LayerNorm()(x)      # critic_layer_norm=True
    x = nn.relu(x)
```
高 UTD 下 critic 梯度规模放大 20 倍，hidden activation 的 scale 会漂移导致 Q 爆炸；LayerNorm 把每层输入归一到单位方差，**切断 scale 漂移的传播链**。RLPD 论文明确这是高 UTD 稳定的必要条件（我们 config 里 `critic_layer_norm=True`，且审查过它的接线是否真的生效——LN 只在 hidden 层包括 activate_final 层，输出 Dense(1) 前不含）。

### Q: 温度（entropy 系数）怎么自动调？target_entropy 为什么是 −dim/2？
```python
temp_loss = α·(entropy − target_entropy)      # entropy = −log π(a|s).mean()
target_entropy = −action_dim / 2              # RLPD 特有！SAC 标准是 −dim
```
SAC 标准把目标熵设为 −dim（鼓励高熵探索）；**RLPD 用 −dim/2**——有离线数据兜底时不需要那么强的探索，熵目标减半让策略更早收敛到离线数据指示的分布。α 的 loss 对 log_alpha 的梯度就是 (entropy − target)（线性），所以 entropy 爆炸时 α 会被梯度猛推——**这也是本次猝死链里"α 崩塌"的机制**（entropy=1e24 → log_alpha 梯度 1e24 量级 → Adam 大步长 → α→0）。

### Q: target network 怎么更新？
`optax.incremental_update(critic.params, target.params, tau=0.005)` = `old + tau·(new−old)` 的 tree_map 版本，每步 critic 更新后 polyak 一次。target_critic 的 TrainState 用空优化器（`GradientTransformation(lambda _: None, lambda _: None)`）——**结构对齐 TrainState 但不走梯度**，这样 ckpt 序列化时结构一致。

## 2. tanh-normal 分布：完整数学 + 本次踩的深坑

### Q: tanh-normal 的 log_prob 公式与数值稳定形式怎么推？
x = tanh(y)，y ~ N(μ, σ²)。换元：
```
log p(x) = log p(y) − Σᵢ log|dx/dyᵢ|
log|dx/dy| = log(1 − tanh²(y)) = log sech²(y)
```
sech²(y) = 4/(eʸ+e⁻ʸ)²，而 log(eʸ+e⁻ʸ) = y + softplus(−2y)，所以：
```
log sech²(y) = 2log2 − 2y − 2·softplus(−2y)
```
代码即 `log_det = Σ 2·(log2 − y − softplus(−2y))`。**为什么不用 1−tanh²(y) 直接算**：|y|>9 时 f32 下 tanh(y)=1.0，1−1.0=0 → log(0)=−inf（灾难性消去）；softplus 形式在任何 y 下都有限。

### Q: 重参数化怎么做？为什么必须？
`y = μ + σ⊙ε, ε~N(0,I)`，sample 返回 `tanh(y)`——ε 显式出现，∂sample/∂μ 和 ∂sample/∂σ 是解析路径，actor loss（α·logπ − Q）对 actor 参数的梯度才能回传。若用非重参数化采样（stop-gradient），actor loss 退化成只有 Q 项的贪心梯度，熵正则失效。

### Q: log_std 为什么 clip 在 [−20, 2]？
- 下限 −20：σ=e⁻²⁰≈2×10⁻⁹。再小则 log_prob 里 z=(y−μ)/σ 溢出 f32；同时 σ→0 让策略退化为确定性（熵→−inf）。
- 上限 2：σ=e²≈7.4，tanh 后动作几乎均匀分布，再大无意义且数值不稳。
- clip 本身在 JAX 里梯度行为 = 区间内恒等、区间外梯度 0——这既保护数值，也意味着 log_std 被 clip 死时对应维度的探索能力**被冻结**。

### Q: 本次最深的坑——tfp 的"缓存语义"到底是什么？
官方 tfp 的 `TransformedDistribution` 对象内部有 `_cache`：**同一对象上先 `sample()` 再 `log_prob(采样值)`，log_prob 会复用 sample 时缓存的中间量 y（精确值），不走 atanh 反推**。RLPD 代码恰好就是这个模式（`actions = dist.sample(seed); log_probs = dist.log_prob(actions)`）。
我们的初版实现从 x 反推：`y = atanh(clip(x, ±(1−1e-6)))`，两条路径在正常区数值一致（所以初版对拍通过），但**饱和区**（σ 触底 + |μ| 偏离）分道扬镳：
- 精确 y 路径：log_prob ≈ 83~653，梯度范数恒定 3.87
- atanh+clip 路径：log_prob ≈ −1.9e10 ~ −3e21，梯度 2e10 ~ **inf**
机理：σ 触底时 z=(y−μ)/σ 分母 2e-9，而 atanh(clip) 的 y 被 clip 在 ±7.25 附近、μ 却可以是 ±千级（actor 爆炸后）→ z² ~ 1e19+ → 求和溢出。
**后果链**：log_prob 天文负 → entropy=−logπ 爆到 1e24 → 温度梯度爆炸 → α 崩塌至 ~0 → 熵正则失效 → actor 纯贪心追 Q → μ 爆炸（actor 参数范数 17→3117）→ 恒定饱和动作 → 分数恒定 ~0。**修复 = `_y_cache`**：sample 时存 y，log_prob 检测到同 shape 输入即走缓存路径（外部数据仍走 atanh）。修复后饱和 log_prob = 85.25。

## 3. JAX 函数式编程的实战要点（面试高频）

### Q: JAX 的 jit 下随机数和 PyTorch 有什么本质区别？
JAX 是**显式函数式 PRNG**：随机 key 是显式参数，`key, rng = jax.random.split(self.rng)` 用掉一个子 key 后必须把新 rng `replace` 回 agent dataclass（`return self.replace(..., rng=rng)`），否则下次 update 用重复 key → 采样完全相关 → 训练静默退化。PyTorch 是全局生成器状态。本项目 agent 是 flax.struct.dataclass，rng 是它的一个 pytree 字段，随 ckpt 一起序列化——**resume 后随机流也从 ckpt 接续**（这是断点续训后训练动态一致的前提之一）。

### Q: 为什么 static_argnames 里要放 utd_ratio 和 q_clip？
jit 只能对**数组**做 tracer 化；Python int/float 会被当常量。`utd_ratio` 控制 Python 层 for 循环次数（编译期展开 20 份 update_critic）、`q_clip` 控制 clip 分支是否存在——二者必须 static。代价：改 utd_ratio 会触发重新编译（每个 (utd_ratio, q_clip) 组合一份 XLA 编译缓存）。

### Q: jnp.clip 的梯度语义是什么？在哪踩过坑？
`jnp.clip(x, lo, hi)` 在区间内梯度 1，**触界处梯度 0**。两处影响：
- log_std 的 clip（[−20,2]）：触底后该维 σ 停止接收梯度（探索冻结）。
- 旧版 tanh_normal 的 `clip(value, ±(1−1e-6))`：动作饱和时 atanh 输入被 clip → **熵项梯度整体断流**，actor 只剩 −Q 的贪心梯度——这是猝死链条的加速器之一。

### Q: DatasetDict 的采样与 combine 怎么实现无 bug 拼接？
```python
def combine(a, b):
    return {k: np.concatenate([a[k], b[k]]) if not isinstance(v, dict) else combine(a[k], b[k])}
```
嵌套字典递归 concat；所有字段 dtype 对齐 float32（single_precision wrapper 处理）。UTD 切片时 assert `x.shape[0] % utd_ratio == 0`——5120/20=256 严格整除，否则 jit 内 assert 直接报错。

## 4. 断点续训与 orbax 的机制细节

### Q: flax checkpoint 的 restore 为什么需要 target？
`checkpoints.restore_checkpoint(ckpt_dir, target=agent, step=None)`：orbax 存的是 pytree 叶子（参数数组），restore 需要一个**结构模板**把数组装回去。target=agent（flax dataclass）→ 按 dataclass 字段名对位还原为同构 dataclass，包括 TrainState（params + opt_state + step）。**叶级对拍**：我们做过 36 路径 restore 前后零差异验证，证明 restore 链路无罪——排障时先排除基础设施再怀疑算法。

### Q: resume 的语义边界是什么？
```python
_latest = checkpoints.latest_checkpoint(chkpt_dir)      # 取 step 最大的
agent = restore_checkpoint(dir, target=agent, step=None) # None = 最新
resume_step = int(str(_latest).rsplit("_", 1)[1])         # 路径名解析 step
for i in tqdm(range(resume_step, max_steps + 1)): ...
```
**能恢复**：agent 参数+优化器状态+RNG 流。**不能恢复**：在线 replay buffer 的已采数据（重启后 buffer 只剩离线预灌 + 新采样本）——RLPD 的对称采样在恢复初期 online 侧分布有偏，NB2 已声明。这个"不可恢复"也是 130k 绕窗策略能工作的**辅助变量之一**（续训 run 的 buffer 状态与整跑 run 不同——死亡窗机制未明的候选解释）。

### Q: keep=20 的滚动删除怎么工作？有什么坑？
`save_checkpoint(dir, agent, step=i, keep=20, overwrite=True)`：保留最近 20 个 step 的 ckpt（5k 步粒度 = 覆盖最近 100k 步）。坑：**旧 ckpt 被滚动删除后，比它更早的死亡现场就"没有物证"了**（我们 15 次猝死的 50k 步以下 ckpt 全部被淘汰，只能靠 wandb 曲线反推）；且 ckpt 是 ~/log_dir 全局 20 个，跨段重跑会覆盖前段物证。

## 5. 工程事故深潜（每个都能讲 10 分钟）

### Q: 130k 步确定性死亡窗——现象、排查路径、对策
- **现象**：从 0 起跑的 run 全部在 126~131k 步 `Fatal Python error: Segmentation fault`（11/11，跨 halfcheetah/hopper/walker2d 跨 seed；faulthandler 栈顶一致指向 update 段 `train_finetuning.py:203`）；从 @125k ckpt 续训的 run 2/2 存活。
- **两个影子**：死亡步数窗口宽 4k（非精确点）→ 排除"某个 eval/save 轮次的精确触发"；墙钟 30~33 分钟 → 墙钟与步数互为影子（速率稳定时不可区分）。
- **排除清单**：数据集（h5 验尸全对）、代码审查（与官方逐行一致）、terminal mask、GPU 风暴（dmesg 无记录）、并发（串行也死）、ckpt 链路（对拍零差异）。
- **未解**：疑似 JAX 0.8/ROCm 7 栈内长运行累积性问题（VRAM 碎片/编译缓存）。
- **对策**：两段式绕窗——第一段 max_steps=125000（窗口前停车），第二段 resume 到目标。数学上：死亡与"从 0 起跑的步数"绑定 → 第二段从 125k 起等效步数仅 75k/125k，整个窗口被结构性跳过。
- **面试加分点**：承认机制未明但给出 11/11 vs 2/2 的统计边界，以及"为什么这是工程上正确的决策"（期望损失：绕窗零代价；不绕窗：100% 死亡）。

### Q: GPU 初始化风暴是什么？为什么错峰救不了训练进程？
MI300X 的 GPU 命令处理器（runlist）在多进程同时做 XLA/TF 初始化时过订阅 → 批量 segfault（dmesg 时间窗可定性）。实测：4 路评估进程 90s 错峰可活；**训练进程**（初始化更重：JAX 编译+GPU 上下文+TF）90s 与 180s 错峰都全灭；且**正在训练的进程会被新进程的 init 撞死**（C@22:07 init → A@22:09 segfault，时间强相关）。结论：一波启动（错峰）后**零新进程**。对策不止错峰：`setsid` 脱离会话、监控循环独立、队列脚本幂等。

### Q: wandb 0.30 离线文件怎么解析的？
wandb 0.30 移除了 Python datastore API，离线文件是 **protobuf 记录序列 + varint 长度前缀**的裸字节流。解析器（tools/q_curves_extract.py）：
1. 逐记录扫字节锚点：`\x12\x11evaluation/return\x82\x01`（0x12=field tag/wire type、0x11=字符串长度 17、`82 01`=值字段 varint tag）
2. 锚点后 1 字节是**载荷长度**（值可能不是定长 float——初版假设固定 4 字节会切碎记录）
3. 与 `\x12\x05_step` 锚点按**字节偏移顺序配对**（value 与其前方最近的 step 绑定）
4. 同一 step 会出现双份（history + summary record），合并去重
5. env/seed 从命令行字节段提取（`--env_name=...` 的明文 grep）
坑：贪婪正则 `[\x20-\x7e]` 会吞掉命令行尾 → 用 `[\x21-\x7e]`；同 run 多段文件要合并而非覆盖。

### Q: 在队列运行中 sed 脚本，为什么会坏？
bash 执行脚本时**按字节偏移逐行读取**（它记住文件内偏移量）。sed 重写文件改变了后续行的字节位置 → bash 从旧偏移读新内容 → 读到"半行 + 下一行开头"的拼接，**max_steps=200000 被读成 180000/75000/70000/65000 等任意损坏值**，`--checkpoint_model=True` 可能被截掉。后果：5 个 run 假完成（DONE_0 但步数不满）。**正确姿势**：要么 `exec` 新脚本文件，要么 `sed ... > new && mv`（原子替换 inode），要么用环境变量传参。验证清单也随之升级：**DONE_0 + 分数平台 ≠ 完成，必须核对 tqdm 分母、ckpt 终点、曲线尾部三证**。

### Q: TF 和 JAX 混装为什么会 segfault？
flax 的 `checkpoints` 模块 import 时拉起 orbax，orbax 初始化 TF runtime——若 TF 不是最先 import 的，XLA/tensorflow_framework 的初始化顺序冲突直接 abort（dmesg 定位到 libtensorflow_framework.so）。**修复一行**：`import tensorflow` 放到所有 import 最前。12/12 终评 segfault 全灭 → 修复后全过。teardown 阶段的 abort（rc=134）无害——json 已落盘。

## 6. 数据与评估口径的细节

### Q: d4rl 的 normalized score 怎么算？v0 和 v2 有什么坑？
`get_normalized_score(return) = (return − random_REF) / (expert_REF − random_REF)`，×100 对齐论文。**坑**：v0 的 REF 表是随机/专家策略在 v0 环境的 return，我们下载的数据集是无版本后缀文件（第三方镜像），若实际是 v2 数据配 v0 REF 会系统性偏移——所以做了 data_sanity.py（h5 轨迹 return 分布 vs 官方画像：hopper-expert norm 108.5，说明数据-REF 兼容）。
另一个口径坑：**TimeLimit.truncated**——gym 的 done 分"真终止（摔倒）"和"超时截断"。Q-learning 的 bootstrap 只在真终止时截断：
```python
mask = 1.0 if (not done or "TimeLimit.truncated" in info) else 0.0
target_q = r + γ·mask·next_q          # mask=0 仅在真终止
```
若把 timeout 也当终止，1000 步长 horizon 的任务（halfcheetah）会系统性低估 Q。

### Q: 为什么 halfcheetah 的数据里有 term%=0？
halfcheetah 物理上不会摔倒（无 terminate 条件），expert 数据 1M 步 terminals 全 0，只有 timeouts。这让它成为"mask 语义 bug 的天然免疫者"——本次排查中 halfcheetah 独活曾一度指向 mask bug，后经逐行审查排除（mask 链路正确）。

## 7. 可以反问面试官的亮点素材

1. "我们用 wandb 的 protobuf 裸字节流逆向了 0.30 移除的解析 API，靠它拿到了 q 值突跳的现场证据"——体现底层功底。
2. "分布库替换的对拍我踩了'正常区一致≠等价'的坑，现在我的对拍清单固定是 三区×两维"——方法论可迁移。
3. "130k 死亡窗机制未明，但我们用 11/11 vs 2/2 的统计边界做了两段式绕窗，把不可解问题变成可调度问题"——工程判断力。
4. "单进程串行实测吞吐反超 4 路并发"——不是教条，是本机实测。
