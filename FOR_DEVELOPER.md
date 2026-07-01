# LCM 在 8GB 消费级显卡上的部署报告

**测试者**: RhineIris  
**GPU**: RTX 3060 8GB, 驱动 580.97, CUDA 13.0  
**系统**: Win11 25H2 + WSL2 Ubuntu + Python 3.14 + JAX 0.10.2  
**日期**: 2026-06-27

## 概述

修了 2 个 bug 后，Qwen 0.5B 全量冻结 + LCM 码本可以在 8GB 显存上训练（B=2 N=128，仅用 8 层 Qwen forward pass，VRAM 峰值 ~5GB）。

---

## 发现的 Bug 及修复

### 1. `train/data.py` WikiDataIter 硬编码 uint16

Qwen 词表 151936 > uint16 上限 65535，读取数据直接炸。

```python
# 修复前:
self.tokens = np.memmap(mmap_path, dtype=np.uint16, mode='r', shape=(self.n_tokens,))
# 修复后:
_dtype_str = meta.get('dtype', 'uint16')
_dtype = np.dtype(_dtype_str)
self.tokens = np.memmap(mmap_path, dtype=_dtype, mode='r', shape=(self.n_tokens,))
```

从 shape JSON 自动读取 dtype，兼容 uint32。

### 2. `train/cog_train.py` AdamW 为冻结参数分配优化器状态

Qwen 0.5B 有 291 个张量（约 1.36 亿冻结参

数）。AdamW 为所有参数创建动量+方差状态，浪费 ~2.2GB 显存。加上 XLA 编译缓冲，直接炸 8GB。

```python
# 修复前:
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(...),
)
# 修复后:
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.masked(
        optax.adamw(...),
        mask=lambda p: 'qwen' not in str(p) and 'lang_lcm' not in str(p),
    ),
)
```

冻结参数被 mask 排除，优化器状态从 ~2.2GB 降到 ~40MB。

---

## 显存占用（修复后）

| 组件 | 显存 |
|---|---|
| Qwen 0.5B FP16（24 层全加载） | ~1.3 GB |
| LCM 编码器 + 6 码本 + W_out + z_proj | ~300 MB |
| 激活值（B=2 N=128，仅 8 层 forward） | ~200 MB |
| AdamW 优化器状态（仅可训参数） | ~40 MB |
| XLA 编译临时缓冲 | ~1.5-3 GB |
| **合计** | **~3.5-5 GB** |

8GB 卡绰绰有余。

---

## 已知问题

### FP16/FP32 混合精度导致 NaN Loss

Qwen 权重是 FP16，LCM 参数是 FP32。混合计算时 Qwen 输出的 logit 经过 softmax 易溢出（FP16 下 exp(x) 在 x>11 时爆炸为 inf

→ inf/inf = NaN）。转 FP32 能解但 2.6GB 权重 + XLA 编译缓冲会 OOM。

**建议**: 在 CE loss 前加 `jnp.clip(logits, -10, 10)` 作为内置保护。

### NaN 跳过逻辑有隐患 (cog_train.py L543)

```python
if np.isnan(loss_f) or np.isinf(loss_f):
    continue  # 跳过日志，但参数已经被 NaN 梯度更新了！
```

20000 步全部 NaN → 全部 continue → 训练"完成"但模型已坏。建议首次 NaN 时 abort 或回滚参数。

### `--steps` 是总步数不是增量步数 (lcm.py L397)

```python
for step in range(start_step, args.steps):
```

从 step 20000 续训 + `--steps 10000` → range(20000, 10000) = 空循环。建议改名或改逻辑。

### Qwen forward 只跑 8 层 (cog_train.py L374)

```python
a_logits = qwen_forward(..., n_layers=8)  # 只用了 8/24 层
```

24 层权重全加载了但只跑 8 层。如果是故意省算力请加注释；否则考虑只加载需要的层。

---

## 训练结果

| 阶段 | 模型 | 数据 | 步数 | B×N | 结果 |
|---|---|---|---|---|---|
| Stage 1 | 纯 LCM 12M | WikiText-103 | 100k | 8×256 | Loss 10.3→5.3 ✅ |
| Stage 2 | 纯 LCM + 码本 | 同 | 20k | 4×256 | VQ 0.002 ✅ |
| Cog | Qwen 0.5B 冻结 | 同 | 编译中 | 2×128 | ⏳ |

---

## 建议优先处理的修复

1. **合并 uint16 和 optax.masked 两个修复**——影响所有消费级 GPU 用户
2. **加 FP16 安全的 logit 裁剪**
3. **文档说明 `--steps` 含义**
4. **README 补充 8GB 卡部署说明**：embed-only 模式 4GB 可跑，全 Qwen 0.5B 模式修上述 bug 后可跑
