import logging

from src.data.etl import load_etl_module


def target_groups() -> dict:
    """Groups of pretraining targets, defined next to the code computing them (etl/aml/targets.py, needs polars)"""
    return load_etl_module("targets").TARGET_GROUPS


def target_names(groups, available, native: bool = False) -> list:
    """Targets of the selected groups that are present in the pretraining data, in group order"""
    targets = load_etl_module("targets")
    allowed = targets.NATIVE_GROUPS if native else targets.USD_GROUPS
    unknown = [group for group in groups if group not in allowed]
    if unknown:
        raise ValueError(f"Target groups {unknown} are not available {'without' if native else 'with'} conversion")

    expected = targets.target_columns(groups, native)
    missing = [name for name in expected if name not in available]
    if missing:
        logging.warning(f"Targets missing from the pretraining data, skipped: {missing}")
    gfp = [name for names in targets.GFP_PATTERNS.values() for name in names]
    extra = [name for name in targets.target_columns(groups, native, gfp_params={"lc-cycle": True}) if name in gfp]
    return [name for name in expected + [name for name in extra if name not in expected] if name in available]


def target_weights(groups, names, group_weights) -> list:
    """Loss weight of every target: the weight of its group (1 if ssl.group_weights is empty)"""
    if group_weights and len(group_weights) != len(groups):
        raise ValueError("ssl.group_weights needs one weight per group in ssl.target_groups")
    weight_of = dict(zip(groups, group_weights or [1.0] * len(groups)))
    group_of = {name: group for group in groups for name in target_groups()[group]}
    return [float(weight_of[group_of[name]]) for name in names]
