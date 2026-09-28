import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed every RNG the training stack draws from: Python, NumPy, and torch on CPU and CUDA.

    This is the same set gs.init(seed=...) seeds. It is called after gs.init because gs_env
    cannot be imported before Genesis is initialized, and the seed is only known once the
    config is parsed. Genesis kernels draw no random numbers, so nothing else needs seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_scalar(value: Any) -> Any:
    """Best-effort parse of CLI string into bool/int/float, else return as-is."""
    if isinstance(value, int | float | bool):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except ValueError:
            return value
    return value


def strip_prefixes(path: str, prefixes: tuple[str, ...]) -> str:
    # repeatedly strip any matching prefix until none match
    changed = True
    while changed:
        changed = False
        for pref in prefixes:
            if path.startswith(pref):
                path = path[len(pref) :]
                changed = True
    return path


def tokenize_path(path: str, prefixes: tuple[str, ...]) -> list[str]:
    """Split a dot/bracket path into tokens after removing known prefixes."""
    path = strip_prefixes(path, prefixes)
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if buf:
                out.append(buf)
                buf = ""
            i += 1
            continue
        if c == "[":
            if buf:
                out.append(buf)
                buf = ""
            j = path.find("]", i)
            if j == -1:
                buf += c
                i += 1
            else:
                out.append(path[i + 1 : j])
                i = j + 1
            continue
        buf += c
        i += 1
    if buf:
        out.append(buf)
    return out


def parse_sequence_like(raw: Any, target_example: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    parts = [p.strip() for p in raw.split(",")]
    parsed = [parse_scalar(p) for p in parts]
    if isinstance(target_example, tuple):
        return tuple(parsed)
    return parsed


def deep_apply(obj: Any, tokens: list[str], raw_val: Any) -> Any:
    """Deeply apply a single override value addressed by tokens to a Pydantic model.

    Handles nested BaseModel, dicts (up to two levels, e.g. reward_args.<Term>.scale),
    and lists/tuples (by index).
    """
    from pydantic import BaseModel as _BM  # type: ignore

    if not tokens:
        return obj

    field = tokens[0]
    rest = tokens[1:]

    # getattr guard
    try:
        current = getattr(obj, field)
    except AttributeError:
        return obj

    # Leaf assignment
    if not rest:
        new_value = parse_scalar(raw_val)
        if isinstance(current, tuple | list) and isinstance(raw_val, str) and "," in raw_val:
            new_value = parse_sequence_like(raw_val, current)
        if isinstance(current, tuple) and isinstance(new_value, list):
            new_value = tuple(new_value)
        return obj.model_copy(update={field: new_value}) if isinstance(obj, _BM) else obj

    # Dict field (e.g., reward_args)
    if isinstance(current, dict):
        key = rest[0]
        new_dict = dict(current)
        if len(rest) == 1:
            new_dict[key] = parse_scalar(raw_val)
        elif isinstance(new_dict.get(key), dict):
            sub = dict(new_dict[key])
            sub_key = rest[1]
            if len(rest) == 2:
                sub[sub_key] = parse_scalar(raw_val)
            else:
                # deeper-than-two-level dict nesting still falls back to a joined key
                sub[".".join(rest[1:])] = parse_scalar(raw_val)
            new_dict[key] = sub
        else:
            new_dict[".".join(rest)] = parse_scalar(raw_val)
        return obj.model_copy(update={field: new_dict})

    # List/Tuple field
    if isinstance(current, list | tuple):
        try:
            idx = int(rest[0])
        except ValueError:
            return obj
        if idx < 0 or idx >= len(current):
            return obj
        seq_list = list(current)
        elem = seq_list[idx]
        if len(rest) == 1:
            new_elem = parse_scalar(raw_val)
        else:
            if hasattr(elem, "model_copy"):
                new_elem = deep_apply(elem, rest[1:], raw_val)
            elif isinstance(elem, dict):
                new_elem = dict(elem)
                new_elem[rest[1]] = parse_scalar(raw_val)
            else:
                new_elem = elem
        seq_list[idx] = new_elem
        new_seq = tuple(seq_list) if isinstance(current, tuple) else seq_list
        return obj.model_copy(update={field: new_seq})

    # Nested Pydantic model
    if hasattr(current, "model_copy"):
        updated_sub = deep_apply(current, rest, raw_val)
        return obj.model_copy(update={field: updated_sub})

    return obj


def apply_overrides_generic(
    base_obj: Any, overrides: dict[str, Any] | None, prefixes: tuple[str, ...]
) -> Any:
    if not overrides:
        return base_obj
    updated = base_obj
    for raw_key, raw_val in overrides.items():
        tokens = tokenize_path(raw_key, prefixes)
        updated = deep_apply(updated, tokens, raw_val)
    return updated


def _convert_to_serializable(obj: Any) -> Any:
    """Convert objects to YAML-serializable format.

    Args:
        obj: Object to convert

    Returns:
        YAML-serializable representation
    """
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuple to list but mark it with a special tag
        return {"__tuple__": [_convert_to_serializable(item) for item in obj]}
    else:
        # For other types, convert to string
        return str(obj)


def _restore_tuples(obj: Any) -> Any:
    """Restore tuples from YAML data.

    Args:
        obj: Object loaded from YAML

    Returns:
        Object with tuples restored
    """
    if isinstance(obj, dict):
        if "__tuple__" in obj and len(obj) == 1:
            # This is a tuple marker
            return tuple(_restore_tuples(item) for item in obj["__tuple__"])
        else:
            return {k: _restore_tuples(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_tuples(item) for item in obj]
    else:
        return obj


def config_to_yaml(config: Any, yaml_path: str | Path) -> None:
    """Convert a config/args class (typically Pydantic model) to YAML file.

    Args:
        config: Configuration object (Pydantic model or similar)
        yaml_path: Path where YAML file should be saved

    Example:
        >>> config_to_yaml(env_args, "logs/<exp_name>/<timestamp>/configs/env_args.yaml")
    """
    from pydantic import BaseModel as _BM  # type: ignore

    # Convert to dictionary
    if isinstance(config, _BM):
        config_dict = config.model_dump()
    elif hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f"Cannot convert config of type {type(config)} to YAML")

    # Convert to serializable format
    serializable_dict = _convert_to_serializable(config_dict)

    # Ensure parent directory exists
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(serializable_dict, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"Config saved to YAML: {yaml_path}")


def yaml_to_config(yaml_path: str | Path, config_class: type | None = None) -> Any:
    """Convert YAML file back to config/args class.

    Args:
        yaml_path: Path to YAML file
        config_class: Optional class type to instantiate. If provided and it's a
                     Pydantic model, will return an instance of that class.
                     If None, returns a dictionary.

    Returns:
        Config object (Pydantic model instance) or dictionary

    Example:
        >>> from gs_env.sim.envs.config.schema import EnvArgs
        >>> env_args = yaml_to_config("logs/<exp_name>/<timestamp>/configs/env_args.yaml", EnvArgs)
    """
    from pydantic import BaseModel as _BM  # type: ignore

    # Load YAML file
    with open(yaml_path, encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Restore tuples
    config_dict = _restore_tuples(config_dict)

    # If config_class provided and is Pydantic model, instantiate it
    if config_class is not None and issubclass(config_class, _BM):
        return config_class(**config_dict)
    elif config_class is not None:
        # Try to instantiate with the dictionary
        return config_class(**config_dict)
    else:
        # Return as dictionary
        return config_dict
