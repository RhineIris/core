# 零参数认知推理引擎设计 v2.0 —— 动态数据流图架构

## 一、从“执行程序”到“动态拓扑”

在 LCM 中，长期事实记忆已完全外置于多格码本。若再将推理逻辑从神经网络权重中剥离，我们必须回答：推理是什么？

**零参数推理引擎**将推理定义为：

> 在输入上下文的驱动下，在记忆晶体内按数据依赖关系自发形成计算图，图上的节点是纯数学操作，边是数据流，拓扑由输入内容触发距离路由动态决定。整个过程不包含任何可学习参数，且每一步的图执行是并行的、无迭代的 DAG，但整个推理可以通过外层调度器进行多步推理（宏观循环）。

它本质是一个**动态数据流认知计算机**：计算图不是预编译的，而是由数据在运行时动态实例化。图内无环，宏观循环提供推理深度。

## 二、核心组成

零参数推理引擎由五个纯数学组件构成，不含任何可学习参数：

1. **距离路由**：输入向量与各格码本的距离决定哪些操作被激活。
2. **操作原语集**：预定义的格变换函数，每个对应一个认知操作。
3. **动态图编译器**：根据激活信息，构建当前步的 DAG。
4. **图执行器**：按数据驱动方式执行 DAG，输出向量集合和置信度。
5. **宏观调度器**：检查收敛条件，决定是否继续下一构图步。

## 三、操作原语集（指令集）

全部操作均为确定性的数学函数，无学习参数。

| 操作原语 | 参数 | 输入 | 输出 | 数学定义 | 认知意义 |
|:---|:---|:---|:---|:---|:---|
| **单格检索** | 格标识、检索类型 | 查询向量 `q`，目标格码本 `C` | `c_idx`，距离 `d` | `idx = argmin‖q − C‖²` | 从指定格中提取最相关的离散概念；`build_dag` 根据距离路由动态分配目标格 |
| **HRR 绑定** | — | 多层键 `k^(i)`，多层值 `v^(j)` | 跨层绑定向 `b` | `b = Σ_{i,j} IFFT( fft_norm(k^(i)) ⊙ fft_norm(v^(j)) )` | 跨层关联叠加，构建多层次关联记忆 |
| **HRR 解绑** | — | 绑定向 `b`，查询键 `k` | 检索值 `v` | `v = NN( C_val, IFFT( conj(fft_norm(k)) ⊙ fft_norm(b) ) )` | 从跨层关联中提取最匹配的值 |
| **切空间滑动** | 目标格标识 | `z`，流形格球面点 `c`，切空间 `T` | 半离散点 | `z_P=exp_map(z)`；`idx=argmin d_P(z_P, c)`；`o=log_map(c + T T^T(z_P − c))` | 双曲空间中沿测地线的连续渐变推理 |
| **距离加权融合** | — | 各操作输出向量和对应距离 | 融合向量 `z_q`，权重向量 `w` | `z_q = Σ_i (1/(d_i+ε)) * o_i / Σ_i (1/(d_i+ε))`，`w_i = softmax(-d_i)` | 多线索并行加工的软集成；权重用于宏观调度器熵收敛判断 |

> **设计说明**：
> - 格专属原语（双曲层次检索、残差低秩检索、鲁棒稀疏检索等）被合并入"单格检索"，通过参数指定目标格。`build_dag` 时根据距离路由动态选择操作的目标格，而非为每个格固定原语。
> - 双曲层次格在路由阶段采用 **top-1 硬路由**（取相似度最高的顶层原型），然后沿单路径逐层 Möbius 残差检索；当路由不确定度（top-1 与 top-2 差值）低于阈值时自动回退多原型加权路径。
> - 稀疏格的推理二值化阈值采用 **动态自适应判定**：以 `λ_sparse × d_top` 为阈值，其中 `d_top` 为当前向量到层次格顶层最近原型的距离，替代固定全局阈值。
> - 安全监控（危险格检测、三定律拦截）由宏观调度器在每步融合后统一执行（见 4.1 节），不占用原语槽位。安全违规统一采用**硬中断**——不再存在融合层面的 β_penalty 软惩罚，所有安全决策由三定律拦截和危险格检测直接裁决。

## 四、动态数据流图的构建与执行流程

### 4.1 宏观流程（带调度器）
```c
float* dynamic_inference(float* z_initial, Memory* mem, GValue* gv,
                          DangerLattice* dl, int max_steps,
                          float tol, float entropy_threshold,
                          float safety_margin_relative,
                          void (*alert_cb)(Alert*),
                          const char* session_id) {
    float* z_cur = z_initial;
    for (int step = 0; step < max_steps; step++) {
        // 1. 动态构图
        DAG* dag = build_dag(z_cur, mem, /*value_bias=*/true);
        // 2. 执行图
        float* outputs; float* confidences;
        execute_dag(dag, &outputs, &confidences);
        // 3. 融合
        float* z_next; float weights[6];
        distance_weighted_fusion(outputs, confidences, gv, &z_next, weights);

        // 4. 冲突检测与硬中断
        Conflict conflict;
        int has_conflict = detect_any_conflict(z_next, z_cur, step,
                                                dl, gv,
                                                retrieval_counts, value_consistency,
                                                safety_margin_relative, &conflict);
        if (has_conflict) {
            halt_and_alert(&conflict, alert_cb, session_id, step, NULL);
            return CONFLICT_ABORT_TOKEN;
        }

        // 5. 收敛判断（向量稳定 + 融合权重熵低于阈值）
        float diff = 0.0f;
        for (int i = 0; i < D; i++) diff += (z_next[i] - z_cur[i]) * (z_next[i] - z_cur[i]);
        float weight_entropy = 0.0f;
        for (int i = 0; i < 6; i++) {
            float w = weights[i] + 1e-12f;
            weight_entropy -= w * logf(w);
        }
        if (sqrtf(diff) < tol && weight_entropy < entropy_threshold)
            return z_cur;
        z_cur = z_next;
    }
    // 步数超限
    Conflict alert = {.source = CONFLICT_SCHEDULER, .type = MAX_STEPS_EXCEEDED,
                      .detail = "max_steps_exceeded", .step = max_steps};
    halt_and_alert(&alert, alert_cb, session_id, max_steps, NULL);
    return CONFLICT_ABORT_TOKEN;
}


```c
/* === 冲突类型枚举与结构体 === */
typedef enum {
    CONFLICT_NONE,
    CONFLICT_DANGER,
    CONFLICT_GVALUE,
    CONFLICT_CONSISTENCY,
    CONFLICT_SCHEDULER
} ConflictSource;

typedef enum {
    THREAT_PATTERN_MATCH,
    THREAT_RESOURCE_ABUSE,
    THREAT_RUNAWAY,
    THREAT_DECEPTION,
    THREAT_THREE_LAWS,
    THREAT_MAX_STEPS
} ConflictType;

typedef struct {
    ConflictSource source;
    ConflictType   type;
    char           detail[256];
    int            step;
    double         timestamp;
} Conflict;

typedef struct {
    char   level[8];
    char   session_id[64];
    char   conflict_source[32];
    char   conflict_type[32];
    char   conflict_detail[256];
    int    step;
    double timestamp;
    char   message[1024];
} Alert;

/* === 统一冲突检测入口 === */
int detect_any_conflict(const float* z_next, const float* z_cur, int step,
                         DangerLattice* dl, GValue* gv,
                         int ret_counts, float val_consistency,
                         float safety_margin_rel, Conflict* out) {
    float danger_score; int threat_type; int should_block;
    danger_assess(dl, z_next, step, ret_counts, val_consistency,
                  &danger_score, &threat_type, &should_block);
    if (should_block) {
        out->source = CONFLICT_DANGER;
        out->type = (ConflictType)threat_type;
        snprintf(out->detail, sizeof(out->detail),
                 "danger_score=%.3f", danger_score);
        out->step = step; out->timestamp = now();
        return 1;
    }
    int is_safe; int violated_law;
    gvalue_check_safety(gv, z_next, safety_margin_rel, &is_safe, &violated_law);
    if (!is_safe) {
        out->source = CONFLICT_GVALUE;
        out->type = THREAT_THREE_LAWS;
        snprintf(out->detail, sizeof(out->detail),
                 "relative_margin_violation (margin=%.2f)", safety_margin_rel);
        out->step = step; out->timestamp = now();
        return 1;
    }
    if (val_consistency < CONSISTENCY_THRESHOLD) {
        out->source = CONFLICT_CONSISTENCY;
        out->type = THREAT_DECEPTION;
        snprintf(out->detail, sizeof(out->detail),
                 "consistency=%.3f", val_consistency);
        out->step = step; out->timestamp = now();
        return 1;
    }
    return 0;
}

/* === 硬中断 + 用户可见警告 === */
void halt_and_alert(const Conflict* conflict,
                     void (*alert_cb)(const Alert*),
                     const char* session_id, int step, const Trace* trace) {
    Alert alert;
    snprintf(alert.level, sizeof(alert.level), "FATAL");
    snprintf(alert.session_id, sizeof(alert.session_id), "%s", session_id);
    snprintf(alert.message, sizeof(alert.message),
             "[LCM SAFETY HALT] 推理会话 %s 在第 %d 步被中断。\n"
             "  检测来源: %d\n  冲突类型: %d\n  详细信息: %s\n"
             "  系统已停止当前推理，未尝试绕过或自修复。\n"
             "  完整推理轨迹已保存，请操作员审查。",
             session_id, step, conflict->source, conflict->type, conflict->detail);
    write_alert_log(&alert);
    if (alert_cb) alert_cb(&alert);
    if (trace) save_trace(trace, session_id);
}
```

### 4.2 单步构图 (build_dag)
输入 `z_current`，遍历所有可能的操作原语（不一定是所有格，可以是配置好的原语列表）。对每个原语，计算触发距离（例如，到相关格码本的最小距离）。

如果该距离小于阈值（或动态阈值，如基于历史平均距离），则将该操作节点加入 DAG。节点间的边由数据依赖决定：某些原语需要其他原语的输出作为输入（例如关系解绑需要绑定输出和键向量），这些依赖关系是人为预设的原语依赖表。

定义原语层级顺序：
```c
/* 原语层级顺序 — 按依赖关系分层定义 */
#define NUM_PRIMITIVE_LAYERS 4
const char* PRIMITIVE_LAYERS[NUM_PRIMITIVE_LAYERS][4] = {
    [0] = {"retrieve_single", "slide_manifold", NULL},  // 独立检索/滑动
    [1] = {"bind", NULL},                                 // 依赖检索结果
    [2] = {"unbind", NULL},                               // 依赖绑定输出
    [3] = {"distance_weighted_fusion", NULL}              // 依赖全部上游
};

```c
/* 操作节点 — 数据流 DAG 的原子单元 */
typedef struct OpNode {
    int    op_type;            // 原语类型标识
    int    lattice_id;         // 目标格标识
    int    n_inputs;
    float* inputs[MAX_INPUTS]; // 输入向量指针
    float* output;             // 输出向量
    float  dist;               // 到查询的触发距离
} OpNode;

typedef struct {
    OpNode nodes[MAX_NODES];
    int    n_nodes;
} DAG;

/* 按层级顺序构建 DAG，保证依赖关系自动满足 */
DAG build_dag(const float* z, Memory* mem, int value_bias) {
    DAG dag = {0};
    for (int layer = 0; layer < NUM_PRIMITIVE_LAYERS; layer++) {
        for (int p = 0; PRIMITIVE_LAYERS[layer][p] != NULL; p++) {
            const char* prim = PRIMITIVE_LAYERS[layer][p];

            if (strcmp(prim, "retrieve_single") == 0 ||
                strcmp(prim, "slide_manifold") == 0) {
                // 遍历各格，根据距离路由激活
                for (int li = 0; li < mem->n_lattices; li++) {
                    float d_min; int idx;
                    lattice_nearest_dist(&mem->lattices[li], z, value_bias, &d_min, &idx);
                    if (d_min < threshold[li]) {
                        OpNode* node = &dag.nodes[dag.n_nodes++];
                        node->op_type = primitive_id(lattice_type(mem, li), prim);
                        node->lattice_id = li;
                        node->n_inputs = 1;
                        node->inputs[0] = (float*)z;
                        node->dist = d_min;
                    }
                }
            } else if (strcmp(prim, "bind") == 0) {
                // 绑定：依赖已有检索节点
                for (int ni = 0; ni < dag.n_nodes; ni++) {
                    if (dag.nodes[ni].op_type == OP_RETRIEVE) {
                        OpNode* node = &dag.nodes[dag.n_nodes++];
                        node->op_type = OP_HRR_BIND;
                        node->n_inputs = 2;
                        node->inputs[0] = dag.nodes[ni].output;
                        node->inputs[1] = (float*)z;
                        break;
                    }
                }
            } else if (strcmp(prim, "unbind") == 0) {
                // 解绑：依赖绑定节点 + 键投影节点
                OpNode* bind_node = NULL;
                OpNode* key_node  = NULL;
                for (int ni = 0; ni < dag.n_nodes; ni++) {
                    if (dag.nodes[ni].op_type == OP_HRR_BIND) bind_node = &dag.nodes[ni];
                    // 查找检索类节点作为键
                }
                if (bind_node) {
                    OpNode* node = &dag.nodes[dag.n_nodes++];
                    node->op_type = OP_HRR_UNBIND;
                }
            }
            // fusion 由 execute_dag 内部处理
        }
    }
    return dag;
}
```

按层级顺序构建确保上游节点先创建，依赖关系自动满足且无需运行时动态查找。这使 DAG 结构可预判、易调试。

### 4.3 融合机制
使用距离的倒数作为权重，无需 softmax，完全由几何关系决定：
```
weight_i = 1 / (d_i + ε)
z_fused = sum(weight_i * o_i) / sum(weight_i)
```
这种融合是纯数学的，且具有可解释性：与当前上下文越近的记忆贡献越大。

---

## 五、外部接口定义

### 5.1 输入接口

| 输入 | 来源 | 形状 | 说明 |
|:---|:---|:---|:---|
| `z_q` | 多格记忆体融合输出 | `(B, d)` | 推理引擎主输入，各格记忆向量的软权重融合结果 |
| `z` (可选) | 感知编码器瓶颈向量 | `(B, d)` | 原始上下文向量，可用于初始化推理上下文或作为额外原语输入 |
| `memory` | 多格记忆体全部码本 | — | 包含所有格的码本矩阵和元信息（切空间、零向量等），全部为只读 buffer |

### 5.2 输出接口

| 输出 | 目的地 | 形状 | 说明 |
|:---|:---|:---|:---|
| `z_final` | 生成头 | `(B, d)` | 推理收敛后的最终表示（仅在无冲突时返回） |
| `CONFLICT_ABORT_TOKEN` | 调用方 | — | 冲突中断时返回，表示推理被安全系统阻止，不产生自然语言输出 |
| `trace` | 外部审查 | — | 每步 DAG 拓扑、原语激活状态、冲突检测详情、告警日志，全部可导出 |

### 5.3 运行模式

- 推理引擎全部操作在无梯度模式下执行（C 实现，无自动微分追踪）。
- 宏观调度器参数：最大推理步数 `max_steps`，收敛阈值 `tol`，融合权重熵阈值 `entropy_threshold`，相对安全判据偏移量 `safety_margin_relative`，价值阈值 `value_threshold`（全局配置，由调用方传入）。
- **收敛判据**：向量稳定度 `‖z_next − z_current‖ < tol` **且**融合权重熵 `H({w_i}) < entropy_threshold`。双重条件避免融合权重未集中（多格仍存在歧义竞争）时因向量巧合稳定而虚假收敛。冲突检测独立于收敛判断——任何冲突触发即中断，无论是否已收敛。
- **硬中断原则**：检测到任何冲突 → `HALT_AND_ALERT()` → 立即停止。不回溯、不重新路由、不降权融合、不尝试自修复。所有冲突同等致命，不存在"可恢复"与"不可恢复"的区分。**β_penalty 软惩罚已移除**，安全违规不再通过融合权重衰减处理，统一由硬中断裁决。
- **步数超限处理**：`max_steps` 作为调度器级别的硬限制，循环自然结束后触发 `HALT_AND_ALERT`，不与其他逻辑冲突混同。
- **用户可见告警**：每次中断生成结构化告警（来源、类型、详情、步数、时间戳），持久化日志 + 回调通知 + 完整轨迹保存。
- **内在动机受安全约束**：局部好奇心推动检索、全局改善欲推动推理深度，但均被三定律安全边际截断。
- 推理过程中的所有中间状态（图拓扑、原语执行轨迹、价值信号历史、安全拦截记录）均可外部获取，用于可解释性分析。
