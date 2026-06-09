STRATEGY_ALIASES = {
    # Proposal-aligned names.
    "cot_to_lean": "cot_then_lean",
    "direct_lean": "direct_lean",
    "interleaved": "mixed_cot_lean",
    "reference_formalization": "reference_to_lean",
    # Backward-compatible legacy names.
    "reference_to_lean": "reference_to_lean",
    "cot_then_lean": "cot_then_lean",
    "mixed_cot_lean": "mixed_cot_lean",
}


def normalize_strategy_name(name):
    key = (name or "").strip().lower()
    if key not in STRATEGY_ALIASES:
        supported = ", ".join(sorted(STRATEGY_ALIASES))
        raise ValueError(f"Unsupported strategy '{name}'. Supported strategies: {supported}")
    return STRATEGY_ALIASES[key]
