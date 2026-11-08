# Dory 1.0 for the QNN static LLM flow: model.py (the graph), convert_weights.py (checkpoint -> state dict).
#
# One export-time workaround, installed when the llama example imports Dory: to_executorch propagates shapes
# through every delegate by running its original graph on fake tensors. Dory's graphs take the 8-bit KV caches
# straight into matmuls (read in place, model.py), which the CPU refuses for mixed dtypes; a delegate's outputs were
# recorded when it was lowered, so those stand in when the graph cannot run.

import importlib

import torch


def _install_delegate_fake_outputs():
    try:   # the module (the package also exports the operator under the same name)
        ecd = importlib.import_module("torch._higher_order_ops.executorch_call_delegate")
    except ImportError:   # an older torch: ExecuTorch's own copy (exir/delegate.py) runs the graph the same way
        return
    run = ecd.call_delegate_cpu
    if getattr(run, "_dory", False):
        return

    def call_delegate_cpu(lowered_module, *args):
        try:
            return run(lowered_module, *args)
        except RuntimeError:
            if not torch._guards.detect_fake_mode(args):
                raise
            outs = [n for n in lowered_module.original_module.graph.nodes if n.op == "output"][0].args[0]
            vals = [o.meta["val"] for o in (outs if isinstance(outs, (tuple, list)) else [outs])]
            res = [torch.empty(v.shape, dtype=v.dtype) for v in vals]
            return tuple(res) if isinstance(outs, (tuple, list)) else res[0]

    call_delegate_cpu._dory = True
    ecd.call_delegate_cpu = call_delegate_cpu   # the fake-mode implementation calls it by this name


_install_delegate_fake_outputs()
