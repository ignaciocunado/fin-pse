import logging

from src.data.etl import load_etl_module


def target_groups() -> dict:
    """Groups of pretraining targets, defined next to the code computing them (etl/aml/targets.py, needs polars)"""
    return load_etl_module("targets").TARGET_GROUPS


def target_names(groups, available) -> list:
    """Targets of the selected groups that are present in the pretraining data, in group order"""
    groups_to_names = target_groups()
    unknown = [group for group in groups if group not in groups_to_names]
    if unknown:
        raise ValueError(f"Unknown target groups {unknown}, choose from {list(groups_to_names)}")

    names = [name for group in groups for name in groups_to_names[group] if name in available]
    missing = [name for group in groups for name in groups_to_names[group] if name not in available]
    if missing:
        logging.warning(f"Targets missing from the pretraining data, skipped: {missing}")
    return names


def target_weights(groups, names, group_weights) -> list:
    """Loss weight of every target: the weight of its group (1 if ssl.group_weights is empty)"""
    if group_weights and len(group_weights) != len(groups):
        raise ValueError("ssl.group_weights needs one weight per group in ssl.target_groups")
    weight_of = dict(zip(groups, group_weights or [1.0] * len(groups)))
    group_of = {name: group for group in groups for name in target_groups()[group]}
    return [float(weight_of[group_of[name]]) for name in names]
