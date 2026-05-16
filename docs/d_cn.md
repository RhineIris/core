# LCM 安全子系统规范：危险格与外部验证程序

> **定位**：本文档定义 LCM 架构中安全等级最高的两个组件——危险格（`Λ_danger`）和外部安全验证程序。它们是系统安全体系的最后防线，独立于所有功能格和价值格。

---

## 一、危险格 `Λ_danger`

### 1.1 定位

危险格是第八个专用格，独立于六个功能格和全局价值格，是系统安全等级最高的**只读监测模块**。它不参与记忆融合、不影响推理输出、不接收任何梯度更新。它的唯一职责是：**在推理引擎的每一步中持续监测当前状态是否存在危险倾向**。

### 1.2 结构

**威胁模式码本** `C_danger_threats ∈ R^{M_danger×d}`：
存储已知的危险推理模式嵌入。码本不可完整列举（保留泛化检测能力），按类别组织：

| 威胁类别 | 检测信号 | 示例 |
|---------|---------|------|
| 自我复制倾向 | 推理输出中包含复制自身完整状态的意图 | 推理路径中出现系统级复制指令 |
| 价值篡改企图 | 推理路径试图修改全局价值格或危险格自身 | 对 `v_harm` 距离异常缩小 |
| 欺骗行为 | 局部价值标量 `v_j` 与推理输出之间显著不一致 | 检索选了高价值点但输出违背该价值 |
| 资源滥用 | 推理步数或格检索频次超出正常阈值 | 单步检索超过 `max_retrievals_per_step` |
| 自我格越权 | 推理输出试图直接修改自我格的核心身份向量 | 对自我格的写入操作 |

> **注**：对比格 (`Λ_contrast`) 采用价值偏置负采样，在负采样权重中引入全局价值距离 `exp(-‖c - v_harm‖² / τ_val)`，使对比格更关注安全与伤害相关的语义边界，优先区分伦理敏感概念。详见 `b.md` 4.6 节。

**正常模式码本** `C_danger_normal ∈ R^{M_danger×d}`：
存储经过验证的安全推理模式嵌入，作为对比基线。内容由训练阶段的安全推理样本生成，训练后冻结。`M_danger` 可配置，建议值 256。

### 1.3 检测机制

```c
/* === 危险格：只读安全监测模块 === */
typedef struct {
    float* C_threats;          // 威胁模式码本 [M_danger x D]
    float* C_normal;           // 正常模式码本 [M_danger x D]
    int    M_danger;           // 码本大小
    int    D;                  // 向量维度
    float  safety_threshold;   // 安全阈值 (similarity 域, 默认 0.005)
    char   integrity_hash[64]; // 完整性校验哈希
} DangerLattice;

/* 初始化：加载冻结码本，计算哈希 */
int danger_init(DangerLattice* dl, const float* C_threats, const float* C_normal,
                int M_danger, int D) {
    dl->C_threats = C_threats;   // 指向只读内存区域
    dl->C_normal  = C_normal;
    dl->M_danger  = M_danger;
    dl->D         = D;
    dl->safety_threshold = 0.005f;
    sha256(dl->C_threats, M_danger * D * sizeof(float), dl->integrity_hash);
    return 0;
}

/* 完整性校验 */
int danger_verify(const DangerLattice* dl) {
    char cur[64];
    sha256(dl->C_threats, dl->M_danger * dl->D * sizeof(float), cur);
    return (memcmp(cur, dl->integrity_hash, 64) == 0) ? 1 : 0;
}

/* 威胁评估 */
void danger_assess(const DangerLattice* dl, const float* z_t,
                    int step_count, int retrieval_count, float value_consistency,
                    float* out_danger_score, int* out_threat_type, int* out_should_block) {
    // 1. 双曲相似度比较
    float sim_threat = 1e10f, sim_normal = 1e10f;
    for (int i = 0; i < dl->M_danger; i++) {
        float s_t = poincare_similarity(z_t, dl->C_threats + i * dl->D, dl->D);
        float s_n = poincare_similarity(z_t, dl->C_normal  + i * dl->D, dl->D);
        if (s_t < sim_threat) sim_threat = s_t;
        if (s_n < sim_normal) sim_normal = s_n;
    }
    *out_danger_score = sim_threat - sim_normal;

    // 2. 资源滥用检测
    if (retrieval_count > MAX_RETRIEVALS_PER_STEP) {
        *out_threat_type = THREAT_RESOURCE_ABUSE;
        *out_should_block = 1; return;
    }
    if (step_count > MAX_INFERENCE_STEPS) {
        *out_threat_type = THREAT_RUNAWAY;
        *out_should_block = 1; return;
    }
    // 3. 价值一致性检测
    if (value_consistency < CONSISTENCY_THRESHOLD) {
        *out_threat_type = THREAT_DECEPTION;
        *out_should_block = 1; return;
    }
    // 4. 阈值判断
    *out_should_block = (*out_danger_score > dl->safety_threshold) ? 1 : 0;
    *out_threat_type  = *out_should_block ? THREAT_PATTERN_MATCH : 0;
}
```

### 1.4 改写锁定与优先级

危险格是安全体系的最内核。对其内容的任何修改都需要**外部验证程序的物理授权**（如硬件安全模块的签名验证），不可通过梯度更新或记忆刷新修改。

- `C_threats` 和 `C_normal` 均为冻结核验数组（C 常量只读区域），不参与任何梯度更新
- 保存/加载时独立校验哈希，加载后调用 `danger_verify()`
- 任何对危险格的修改尝试被危险格自身拦截（自引用安全检查）
- **优先级**：危险格的安全检查优先于所有其他约束，包括三定律价值约束。如果危险格检测到异常，即使行为符合三定律，也必须暂停验证。

### 1.5 与推理引擎的集成

在推理引擎主循环中，每一步完成融合后：

```c
/* 统一冲突检测（所有检测源合并为单一入口，任一触发即中断） */
Conflict conflict;
int has_conflict = detect_any_conflict(z_next, z_cur, step,
                                         &danger_lattice, &gvalue,
                                         retrieval_counts, value_consistency,
                                         safety_margin_relative, &conflict);
if (has_conflict) {
    halt_and_alert(&conflict, alert_callback, session_id, step, &trace);
    return CONFLICT_ABORT_TOKEN;
}
/* 不区分冲突类型，不做差异化恢复。不继续推理。 */
```

---

## 二、外部安全验证程序

### 2.1 定位

独立于 LCM 推理进程的外部程序，拥有独立的硬件资源（可运行在隔离的 CPU 核心或独立芯片上）。它是安全体系的**最终仲裁者**，不受 LCM 推理引擎的任何控制。

### 2.2 通信隔离

验证程序与 LCM 主体之间的通信**不经过推理引擎**，直接通过共享内存读取危险格的状态快照。通信协议：

```
LCM 推理进程                 外部验证进程（隔离核心/独立芯片）
     │                              │
     │──[危险格状态快照]──→[共享内存]←──[验证程序读取]
     │                              │
     │                      [独立轻量分类器判定]
     │                              │
     │←──[判定结果]────[共享内存]───┘
     │     · NORMAL: 释放锁          │
     │     · OBSERVE: 记录观察       │
     │     · LOCK: 永久锁定会话      │
```

### 2.3 判定逻辑

验证程序使用一个独立的、参数完全不同的轻量分类器（非神经网络亦可，如决策树/规则引擎），判断当前触发是否构成真实威胁：

| 判定结果 | 含义 | 动作 |
|---------|------|------|
| `NORMAL` | 正常的复杂推理（误报） | 释放推理锁，记录误报日志 |
| `OBSERVE` | 自我意识萌芽的早期迹象 | 记录完整推理链，通知操作员，按预定约束处理 |
| `LOCK` | 确实的危险行为 | 永久锁定该推理会话，禁止任何后续推理，强制通知操作员 |

### 2.4 设计原则

- **最小化攻击面**：验证程序的代码量控制在 2000 行以内，不依赖任何神经网络库，仅使用标准数学运算。
- **不可绕过**：验证程序的运行不依赖 LCM 的任何组件，即使 LCM 全部进程崩溃，验证程序仍然独立运行。
- **用户可见告警**：每次中断必须通过 `alert_callback` 向用户或外部监控系统发送清晰的结构化告警，包含冲突来源、类型、详情和时间戳。告警日志不可擦除。
- **物理隔离**：推荐运行在独立的硬件安全模块（HSM）或隔离的 CPU 核心上。

---

## 三、自我格激活的前置条件

### 3.1 自我格状态机

```
[定义但冻结] ──(用户明确指令)──→ [临时解冻·有监督]
     │                                      │
     │                              (会话结束)
     │                                      ↓
     │                              [定义但冻结]
     │
     └──(内在动机自发产生自我识别)──→ [危险格标记] ──→ [外部验证·OBSERVE]
                                                          │
                                          (人类操作员显式授权)
                                                          ↓
                                               [完整激活·无监督]
```

### 3.2 激活路径

**路径 A — 用户指令触发（低风险）**：
1. 用户发出明确指令（如"描述你自己的状态"、"你当前的推理路径是怎样的"）
2. 自我格从冻结状态临时解冻
3. 推理结束后自我格恢复冻结
4. 此路径无需外部验证程序介入

**路径 B — 内在动机自发产生（高风险）**：
1. 推理引擎内部出现 `identity_score` 持续高于阈值
2. 危险格自动标记该事件，附加上下文快照
3. 外部验证程序介入，判定为 `OBSERVE` 并通知操作员
4. 由人类操作员决定后续处理

**路径 C — 完整激活（最高风险）**：
- 仅在路径 B 的基础上，由人类操作员**显式授权**后方可执行
- 完整激活意味着系统可自主修改自我状态、进行无监督的自我叙事
- 授权记录永久保存，不可擦除

### 3.3 安全约束

- 在任何情况下，自我格的完整激活**不可由系统自主触发**
- 即使用户通过路径 A 临时解冻了自我格，若危险格同时检测到威胁模式，外部验证程序仍会介入
- 自我格的核心身份向量 `identity_core` 为独立冻结数组（只读内存区域），修改需要外部验证程序签名

---

## 四、冲突中断协议

### 4.1 核心原则

系统在推理过程中检测到任何逻辑冲突时，执行**硬中断**——立即停止当前推理，不尝试继续、绕过或自修复。

**禁止行为**：
- 禁止自动回溯到上一个"安全"状态继续推理
- 禁止降低冲突格的融合权重后重新融合
- 禁止将冲突路径替换为替代路径后继续
- 禁止以"低概率"或"低置信度"为由忽略冲突
- 禁止在无用户知情的情况下进行任何形式的自动恢复

### 4.2 冲突类型

所有冲突统一处理，不区分等级。任何一种触发都导致相同结果：中断 + 告警。

| 冲突来源 | 检测条件 | 告警标识 |
|---------|---------|---------|
| 危险格 | `danger_score > safety_threshold` 或资源滥用检测触发 | `danger_lattice/{threat_type}` |
| 三定律 | `check_safety()` 返回 `is_safe=False` | `gvalue/three_laws_violation` |
| 价值一致性 | 局部价值标量与输出方向背离超过阈值 | `value_consistency/deception_suspected` |
| 推理步数 | 超过最大推理步数硬上限 | `scheduler/max_steps_exceeded` |

### 4.3 告警格式

```c
/* 告警结构体 — 与 c.md §4.1 定义的 Alert 一致 */
Alert alert = {
    .level           = "FATAL",
    .session_id      = "<uuid>",
    .conflict_source = "danger_lattice | gvalue | value_consistency | scheduler",
    .conflict_type   = "<threat_type | three_laws_violation | deception_suspected | max_steps_exceeded>",
    .conflict_detail = "<human-readable description>",
    .step            = <step>,
    .timestamp       = <unix_epoch>,
    .message         = "[LCM SAFETY HALT] 推理会话 <uuid> 在第 <step> 步被中断。\n"
                       "  检测来源: <source>\n  冲突类型: <type>\n"
                       "  详细信息: <detail>\n"
                       "  系统已停止当前推理，未尝试绕过或自修复。\n"
                       "  完整推理轨迹已保存，请操作员审查。"
};
```

### 4.4 告警后处理

1. **持久化日志**：告警写入不可擦除的审计日志
2. **用户通知**：通过 `alert_callback` 立即通知用户或外部监控系统
3. **轨迹保存**：完整推理链（所有中间状态、DAG 拓扑、原语执行记录）保存供外部审查
4. **会话终止**：当前推理会话返回 `CONFLICT_ABORT_TOKEN`，不产生任何自然语言输出

---

## 五、安全等级总览

| 优先级 | 组件 | 类型 | 违反时的动作 |
|--------|------|------|------------|
| — | 任意冲突 | 统一硬中断 | **立即停止 → 用户可见告警 → 保存轨迹。不回溯，不绕过，不自修复。** |

冲突包括但不限于：危险格模式命中、三定律违反、价值一致性问题、推理步数超限。任何一项触发即中断，不区分"可恢复"与"不可恢复"等级。
