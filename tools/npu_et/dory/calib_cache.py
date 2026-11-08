# Calibration once, reused: the activation observers of a calibrated PT2E graph saved to a file and put back into
# a later export of the same model code (the context length, AR length and sharding may differ: they change shapes,
# not the graph's nodes). Weight and constant observers are not saved: observe_weights shows them their tensors.
#
# llm_wrappers reads XTG_CALIB_SAVE=PATH (save after calibrating) and XTG_CALIB_LOAD=PATH (skip calibration).
# Saved per observer: the node it watches, its class and its state_dict (min_val / max_val and the like; the
# modules themselves hold pybind objects and do not pickle).

import logging

import torch


def _activation_observers(gm: torch.fx.GraphModule) -> dict:
    out = {}
    for node in gm.graph.nodes:
        if node.op != "call_module" or not node.target.startswith("activation_post_process_"):
            continue
        src = node.args[0] if node.args else None
        if getattr(src, "op", None) == "get_attr":   # a weight or a constant
            continue
        out[node.target] = (getattr(src, "name", ""), gm.get_submodule(node.target))
    return out


def observe_weights(gm: torch.fx.GraphModule) -> int:
    """Every observer that watches a weight or a constant sees it, as a forward would show it (the forward itself
    is what this avoids: the calibration graph runs AR = context tokens)."""
    n = 0
    with torch.no_grad():
        for node in gm.graph.nodes:
            if node.op != "call_module" or not node.target.startswith("activation_post_process_"):
                continue
            src = node.args[0] if node.args else None
            if getattr(src, "op", None) != "get_attr":
                continue
            t = gm
            for part in src.target.split("."):
                t = getattr(t, part)
            gm.get_submodule(node.target)(t)
            n += 1
    logging.info("observed %d weights and constants", n)
    return n


def save_activation_observers(gm: torch.fx.GraphModule, path: str) -> None:
    obs = _activation_observers(gm)
    state = {name: (src, type(mod).__name__, {k: v.detach().cpu() for k, v in mod.state_dict().items()})
             for name, (src, mod) in obs.items()}
    torch.save(state, path)
    logging.info("saved %d activation observers to %s", len(obs), path)


def load_activation_observers(gm: torch.fx.GraphModule, path: str) -> None:
    saved = torch.load(path, weights_only=True)
    obs = _activation_observers(gm)
    missing = sorted(set(obs) - set(saved))
    extra = sorted(set(saved) - set(obs))
    if missing or extra:
        raise RuntimeError(f"{path} does not match this graph: {len(missing)} observers missing "
                           f"({missing[:3]}), {len(extra)} unknown ({extra[:3]}); calibrate again")
    for name, (src, mod) in obs.items():
        saved_src, saved_cls, state = saved[name]
        if saved_src != src or saved_cls != type(mod).__name__:
            raise RuntimeError(f"{path}: observer {name} watched {saved_src} ({saved_cls}), "
                               f"here {src} ({type(mod).__name__}); calibrate again")
        for k, v in state.items():   # a buffer's shape can differ from the fresh one (empty before observing)
            getattr(mod, k).resize_(v.shape).copy_(v)
    logging.info("loaded %d activation observers from %s", len(obs), path)
