// xtg_npu: a model of ExecuTorch's Qualcomm static LLM flow (a .pte with a prefill method of AR-P tokens and a
// decode method of AR-D: 1, or a few for the verify steps of speculative decoding; KV caches as graph I/O) run on
// the Hexagon NPU, behind a C API that xtier-go loads with
// dlopen (packs/dory, backend=npu). One handle is one session: its KV caches, the causal and sliding-window masks,
// the position. Logits come out as FP32 rows.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xtg_npu xtg_npu;

// row: index in this call's tokens; logits: vocab floats, valid during the callback
typedef void (*xtg_npu_logits_fn)(int row, const float* logits, void* user);

// Load the .pte (options: JSON, may be empty; "snapshots": ring snapshots kept, default 2; "decode_rows": calls of
// up to this many tokens use the decode method, default 4 for AR-1 and AR-D otherwise, -1 never; "mlock": 1 locks
// the mapped .pte, default 0). NULL on failure, the reason in err.
xtg_npu* xtg_npu_open(const char* pte_path, const char* options, char* err, int err_len);
// {"vocab", "context", "max_position", "prefill_ar", "decode_ar", "kv_instances", "ring_instances", "ring_slots",
//  "sliding_window", "kv_bytes", "snapshots", "logits_scale", "logits_zero_point"} as JSON; the length or -1.
int xtg_npu_info(xtg_npu* h, char* out, int out_len);
// Run n tokens at positions pos() .. pos() + n - 1. want: 0 no logits, 1 the last row, 2 every row. 0 on success.
int xtg_npu_forward(xtg_npu* h, const int32_t* tokens, int n, int want, xtg_npu_logits_fn fn, void* user, char* err,
                    int err_len);
int64_t xtg_npu_pos(xtg_npu* h);
// Back to n positions (n <= pos()). Full-attention caches keep every position; the sliding rings must still hold
// the window before n (a rewind within the ring's slack), else -1 and nothing changes. 0 on success.
int xtg_npu_rewind(xtg_npu* h, int64_t n);
// Ring snapshots for the prefix cache: remember the rings at pos() in a slot, go back to them later (pos() becomes
// the snapshot's). A forward below a snapshot's position drops it.
int xtg_npu_snapshot_slots(xtg_npu* h);
int64_t xtg_npu_snapshot_pos(xtg_npu* h, int slot);   // -1: empty
void xtg_npu_snapshot(xtg_npu* h, int slot);
int xtg_npu_restore(xtg_npu* h, int slot);   // 0 on success
// Move the session to another .pte of the same export with another context length (a bucket that holds pos()):
// the caches, the position and the snapshots it can hold come along, nothing is computed again. The current
// contexts are freed first; on failure the session reopens its own .pte with the same state and returns -1.
int xtg_npu_grow(xtg_npu* h, const char* pte_path, char* err, int err_len);
void xtg_npu_close(xtg_npu* h);

#ifdef __cplusplus
}
#endif
