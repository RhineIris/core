# Zero-Parameter Cognitive Inference Engine Design v2.0 — Dynamic Dataflow Graph Architecture

## 1. From "Executing a Program" to "Dynamic Topology"

In LCM, long-term factual memory has been fully externalized into multi-lattice codebooks. If we further strip reasoning logic from neural network weights, we must answer: what is reasoning?

**Zero-parameter inference engine** defines reasoning as:

> Driven by the input context, a computation graph spontaneously forms within the memory crystal according to data dependencies. The nodes are pure mathematical operations, the edges are data flows, and the topology is dynamically determined by distance routing triggered by the input content. The entire process contains no learnable parameters, and the execution of the graph at each step is a parallel, non-iterative DAG, though the overall inference can proceed through multi-step reasoning (macro loop) via an outer scheduler.

It is essentially a **dynamic dataflow cognitive computer**: the computation graph is not precompiled but dynamically instantiated at runtime by the data. The graph is acyclic, while the macro loop provides reasoning depth.

## 2. Core Components

The zero-parameter inference engine consists of five purely mathematical components, containing no learnable parameters:

1. **Distance Routing**: distances between the input vector and each lattice codebook determine which operations are activated.
2. **Primitive Operation Set**: predefined lattice transformation functions, each corresponding to a cognitive operation.
3. **Dynamic Graph Compiler**: builds the DAG for the current step based on activation information.
4. **Graph Executor**: executes the DAG in a data-driven manner, producing a set of output vectors and confidences.
5. **Macro Scheduler**: checks convergence conditions and decides whether to proceed to the next graph construction step.

## 3. Primitive Operation Set (Instruction Set)

All operations are deterministic mathematical functions with no learnable parameters.

| Primitive | Parameters | Input | Output | Mathematical Definition | Cognitive Significance |
|:---|:---|:---|:---|:---|:---|
| **Single Lattice Retrieve** | lattice_id, retrieval type | query vector `q`, target lattice codebook `C` | `c_idx`, distance `d` | `idx = argmin‖q − C‖²` | Extract the most relevant discrete concept from the specified lattice; `build_dag` dynamically assigns the target lattice based on distance routing |
| **HRR Bind** | — | multi-layer keys `k^(i)`, multi-layer values `v^(j)` | cross-layer binding vector `b` | `b = Σ_{i,j} IFFT( fft_norm(k^(i)) ⊙ fft_norm(v^(j)) )` | Cross-layer association superposition, building multi-level associative memories |
| **HRR Unbind** | — | binding vector `b`, query key `k` | retrieved value `v` | `v = NN( C_val, IFFT( conj(fft_norm(k)) ⊙ fft_norm(b) ) )` | Extract the best-matching value from cross-layer associations |
| **Tangent Space Slide** | target lattice_id | `z`, manifold lattice spherical point `c`, tangent space `T` | semi-discrete point | `z_P=exp_map(z)`; `idx=argmin d_P(z_P, c)`; `o=log_map(c + T T^T(z_P − c))` | Continuous gradual reasoning along geodesics in hyperbolic space |
| **Distance-Weighted Fusion** | — | output vectors from each operation and their corresponding distances | fused vector `z_q`, weight vector `w` | `z_q = Σ_i (1/(d_i+ε)) * o_i / Σ_i (1/(d_i+ε))`, `w_i = softmax(-d_i)` | Soft integration of multi-cue parallel processing; weights used for macro scheduler entropy convergence judgment |

> **Design Notes**:
> - Lattice-specific primitives (hyperbolic hierarchical retrieval, residual low-rank retrieval, robust sparse retrieval, etc.) are consolidated into "Single Lattice Retrieve", with the target lattice specified by parameters. During `build_dag`, the target lattice of an operation is dynamically selected based on distance routing, rather than fixing a primitive for each lattice.
> - The hyperbolic hierarchical lattice uses **top-1 hard routing** during the routing phase (selecting the top-level prototype with the highest similarity), then performs layer-by-layer Mobius residual retrieval along a single path; when routing uncertainty (difference between top-1 and top-2) falls below a threshold, it automatically falls back to a multi-prototype weighted path.
> - The inference binarization threshold for sparse lattices uses **dynamic adaptive determination**: the threshold is `lambda_sparse * d_top`, where `d_top` is the distance from the current vector to the nearest top-level prototype of the hierarchical lattice, replacing a fixed global threshold.
> - Safety monitoring (danger lattice detection, Three Laws interception) is executed uniformly by the macro scheduler after each fusion step (see Section 4.1) and does not occupy primitive slots. Safety violations uniformly adopt **hard abort** -- there is no longer a beta_penalty soft penalty at the fusion level; all safety decisions are directly adjudicated by Three Laws interception and danger lattice detection.

## 4. Construction and Execution of the Dynamic Dataflow Graph

### 4.1 Macro Flow (with Scheduler)
```c
float* dynamic_inference(float* z_initial, Memory* mem, GValue* gv,
                          DangerLattice* dl, int max_steps,
                          float tol, float entropy_threshold,
                          float safety_margin_relative,
                          void (*alert_cb)(Alert*),
                          const char* session_id) {
    float* z_cur = z_initial;
    for (int step = 0; step < max_steps; step++) {
        // 1. Dynamic graph construction
        DAG* dag = build_dag(z_cur, mem, /*value_bias=*/true);
        // 2. Execute graph
        float* outputs; float* confidences;
        execute_dag(dag, &outputs, &confidences);
        // 3. Fusion
        float* z_next; float weights[6];
        distance_weighted_fusion(outputs, confidences, gv, &z_next, weights);

        // 4. Conflict detection and hard abort
        Conflict conflict;
        int has_conflict = detect_any_conflict(z_next, z_cur, step,
                                                dl, gv,
                                                retrieval_counts, value_consistency,
                                                safety_margin_relative, &conflict);
        if (has_conflict) {
            halt_and_alert(&conflict, alert_cb, session_id, step, NULL);
            return CONFLICT_ABORT_TOKEN;
        }

        // 5. Convergence check (vector stable + fusion weight entropy below threshold)
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
    // Max steps exceeded
    Conflict alert = {.source = CONFLICT_SCHEDULER, .type = MAX_STEPS_EXCEEDED,
                      .detail = "max_steps_exceeded", .step = max_steps};
    halt_and_alert(&alert, alert_cb, session_id, max_steps, NULL);
    return CONFLICT_ABORT_TOKEN;
}


```c
/* === Conflict type enum and struct === */
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

/* === Unified conflict detection entry === */
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

/* === Hard abort + user-visible alert === */
void halt_and_alert(const Conflict* conflict,
                     void (*alert_cb)(const Alert*),
                     const char* session_id, int step, const Trace* trace) {
    Alert alert;
    snprintf(alert.level, sizeof(alert.level), "FATAL");
    snprintf(alert.session_id, sizeof(alert.session_id), "%s", session_id);
    snprintf(alert.message, sizeof(alert.message),
             "[LCM SAFETY HALT] Inference session %s aborted at step %d.\n"
             "  Source: %d\n  Type: %d\n  Detail: %s\n"
             "  System has stopped current reasoning without bypass or self-repair.\n"
             "  Full inference trace saved for operator review.",
             session_id, step, conflict->source, conflict->type, conflict->detail);
    write_alert_log(&alert);
    if (alert_cb) alert_cb(&alert);
    if (trace) save_trace(trace, session_id);
}
```

### 4.2 Single-Step Graph Construction (build_dag)

Input `z_current`, iterate over all possible operation primitives (not necessarily all lattices; it can be a configured list of primitives). For each primitive, compute the trigger distance (e.g., the minimum distance to the relevant lattice codebook).

If this distance is less than a threshold (or dynamic threshold, such as one based on historical average distance), the operation node is added to the DAG. Edges between nodes are determined by data dependencies: some primitives require outputs from other primitives as input (for example, unbinding requires the binding output and a key vector); these dependencies are specified in a predefined primitive dependency table.

Define the primitive layer ordering:
```c
/* Primitive layer order — defined by dependency hierarchy */
#define NUM_PRIMITIVE_LAYERS 4
const char* PRIMITIVE_LAYERS[NUM_PRIMITIVE_LAYERS][4] = {
    [0] = {"retrieve_single", "slide_manifold", NULL},  // Independent retrieve/slide
    [1] = {"bind", NULL},                                 // Depends on retrieval results
    [2] = {"unbind", NULL},                               // Depends on binding output
    [3] = {"distance_weighted_fusion", NULL}              // Depends on all upstream
};

```c
/* Op node — atomic unit of the dataflow DAG */
typedef struct OpNode {
    int    op_type;            // Primitive type identifier
    int    lattice_id;         // Target lattice identifier
    int    n_inputs;
    float* inputs[MAX_INPUTS]; // Input vector pointers
    float* output;             // Output vector
    float  dist;               // Trigger distance to query
} OpNode;

typedef struct {
    OpNode nodes[MAX_NODES];
    int    n_nodes;
} DAG;

/* Build DAG in layer order, ensuring dependencies are satisfied automatically */
DAG build_dag(const float* z, Memory* mem, int value_bias) {
    DAG dag = {0};
    for (int layer = 0; layer < NUM_PRIMITIVE_LAYERS; layer++) {
        for (int p = 0; PRIMITIVE_LAYERS[layer][p] != NULL; p++) {
            const char* prim = PRIMITIVE_LAYERS[layer][p];

            if (strcmp(prim, "retrieve_single") == 0 ||
                strcmp(prim, "slide_manifold") == 0) {
                // Iterate over all lattices, activate based on distance routing
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
                // Bind: depends on existing retrieval nodes
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
                // Unbind: depends on binding node + key projection node
                OpNode* bind_node = NULL;
                OpNode* key_node  = NULL;
                for (int ni = 0; ni < dag.n_nodes; ni++) {
                    if (dag.nodes[ni].op_type == OP_HRR_BIND) bind_node = &dag.nodes[ni];
                    // Look for retrieval-type nodes as keys
                }
                if (bind_node) {
                    OpNode* node = &dag.nodes[dag.n_nodes++];
                    node->op_type = OP_HRR_UNBIND;
                }
            }
            // Fusion is handled internally by execute_dag
        }
    }
    return dag;
}
```

Building in layer order ensures that upstream nodes are created first, dependencies are satisfied automatically without runtime dynamic lookups. This makes the DAG structure predictable and easy to debug.

### 4.3 Fusion Mechanism

Uses the reciprocal of distance as weights, no softmax needed, entirely determined by geometric relationships:
```
weight_i = 1 / (d_i + ε)
z_fused = sum(weight_i * o_i) / sum(weight_i)
```
This fusion is purely mathematical and interpretable: memories closer to the current context contribute more.

---

## 5. External Interface Definitions

### 5.1 Input Interface

| Input | Source | Shape | Description |
|:---|:---|:---|:---|
| `z_q` | Multi-lattice memory fusion output | `(B, d)` | Main input to the inference engine, the soft-weighted fusion result of each lattice's memory vectors |
| `z` (optional) | Perception encoder bottleneck vector | `(B, d)` | Raw context vector, can be used to initialize the inference context or as additional primitive input |
| `memory` | All codebooks of the multi-lattice memory | — | Contains codebook matrices and metadata (tangent spaces, zero vectors, etc.) for all lattices, all read-only buffers |

### 5.2 Output Interface

| Output | Destination | Shape | Description |
|:---|:---|:---|:---|
| `z_final` | Generation head | `(B, d)` | Final representation after inference convergence (returned only when no conflict occurs) |
| `CONFLICT_ABORT_TOKEN` | Caller | — | Returned when aborted due to conflict, indicating inference was blocked by the safety system, producing no natural language output |
| `trace` | External audit | — | Per-step DAG topology, primitive activation status, conflict detection details, alert logs -- all exportable |

### 5.3 Operating Modes

- All operations of the inference engine execute in gradient-free mode (C implementation, no automatic differentiation tracing).
- Macro scheduler parameters: maximum inference steps `max_steps`, convergence tolerance `tol`, fusion weight entropy threshold `entropy_threshold`, relative safety margin offset `safety_margin_relative`, value threshold `value_threshold` (global configuration, passed in by the caller).
- **Convergence criterion**: vector stability `||z_next - z_current|| < tol` **and** fusion weight entropy `H({w_i}) < entropy_threshold`. The dual condition prevents false convergence when fusion weights have not consolidated (multi-lattice ambiguity competition still exists) but the vector happens to be stable by coincidence. Conflict detection is independent of convergence judgment -- any triggered conflict causes an abort, regardless of whether convergence has been reached.
- **Hard abort principle**: any conflict detected -> HALT_AND_ALERT() -> immediate stop. No backtracking, no rerouting, no down-weighting fusion, no attempted self-repair. All conflicts are equally fatal; there is no distinction between "recoverable" and "unrecoverable". **beta_penalty soft penalty has been removed** -- safety violations are no longer handled through fusion weight decay, but are uniformly adjudicated by hard abort.
- **Step limit handling**: `max_steps` serves as a scheduler-level hard limit; after the loop ends naturally, HALT_AND_ALERT is triggered, not conflated with other logical conflicts.
- **User-visible alerts**: each abort generates a structured alert (source, type, detail, step, timestamp), with persistent logging + callback notification + full trace saving.
- **Intrinsic motivation bound by safety**: local curiosity drives retrieval, global improvement drive pushes reasoning depth, but both are truncated by the Three Laws safety margin.
- All intermediate states during inference (graph topology, primitive execution traces, value signal history, safety interception records) can be externally accessed for interpretability analysis.
