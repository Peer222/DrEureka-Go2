from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any, Sequence
import tyro
import pickle


@dataclass
class Args:
    run_dir1: Path
    run_dir2: Path


def load_cfg(run_cfg: Path) -> dict:
    if run_cfg.is_dir():
        run_cfg = run_cfg / "parameters.pkl"
    with open(run_cfg, "rb") as f:
        return pickle.load(f)


def compare_lists(l1, l2) -> tuple[Sequence[Any], Sequence[tuple[dict, dict]]]:
    diff = [i for i, j in zip(l1, l2) if i != j]
    if len(l1) < len(l2):
        diff.extend(l2[len(l1) :])
    if len(l1) < len(l2):
        diff.extend(l1[len(l2) :])
    equal_dicts = [
        (i, j) for i, j in zip(l1, l2) if isinstance(i, dict) and isinstance(j, dict)
    ]
    return (diff, equal_dicts)


def compare_cfgs(cfg1: dict, cfg2: dict, level: str = "Cfg"):
    keys1 = set(cfg1.keys())
    keys2 = set(cfg2.keys())
    new_keys1 = keys1.difference(keys2)
    new_keys2 = keys2.difference(keys1)
    shared_keys = keys1.intersection(keys2)

    if len(new_keys1):
        print(f"Only in {level} 1: {list(new_keys1)}")
    if len(new_keys2):
        print(f"Only in {level} 2: {list(new_keys2)}")

    nesting_keys = []
    for key in shared_keys:
        if cfg1[key] == cfg2[key]:
            continue
        if type(cfg1[key]) != type(cfg2[key]):
            print(f"Type mismatch {level}.{key}: {type(cfg1[key])} : {type(cfg2[key])}")
        elif isinstance(cfg1[key], dict):
            nesting_keys.append(key)
        elif isinstance(cfg1[key], Iterable):
            diff, equal_dicts = compare_lists(cfg1[key], cfg2[key])
            if len(diff):
                print(f"List mismatch {level}.{key}: {cfg1[key]} : {cfg2[key]}")
            for i, (dict1, dict2) in enumerate(equal_dicts):
                compare_cfgs(dict1, dict2, f"{level}.{key}.{i}")
        else:
            print(f"Value mismatch {level}.{key}: {cfg1[key]} : {cfg2[key]}")

    for key in nesting_keys:
        compare_cfgs(cfg1[key], cfg2[key], level=f"{level}.{key}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    cfg1 = load_cfg(args.run_dir1)
    cfg2 = load_cfg(args.run_dir2)

    compare_cfgs(cfg1, cfg2)
