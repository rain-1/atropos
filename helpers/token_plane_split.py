"""Utility for constructing a near-even red/blue token split from embedding space.

This script samples random hyperplanes in a model's token embedding space,
then tweaks the bias / boundary assignment so the resulting split is as close to
50:50 as possible (exactly 50:50 for even eligible-token counts).
"""

from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PlaneSplitMetrics:
    eligible_tokens: int
    red_tokens: int
    blue_tokens: int
    red_ratio: float
    blue_ratio: float
    balance_error: float
    attempts_used: int
    tie_count: int


def _normalize_surface(token_text: str) -> str:
    return token_text.replace("▁", " ").strip().lower()


def _is_ignorable_surface(
    token_text: str,
    ignore_punctuation_tokens: bool,
    ignore_whitespace_tokens: bool,
    ignored_token_strings: set[str],
) -> bool:
    norm = _normalize_surface(token_text)
    if ignore_whitespace_tokens and norm == "":
        return True
    if ignore_punctuation_tokens and norm and all(ch in string.punctuation for ch in norm):
        return True
    return norm in ignored_token_strings


def _eligible_token_ids(
    tokenizer,
    ignore_punctuation_tokens: bool,
    ignore_whitespace_tokens: bool,
    ignored_token_strings: Iterable[str],
) -> tuple[list[int], list[int]]:
    special_ids = set(tokenizer.all_special_ids or [])
    ignored_strings = {s.strip().lower() for s in ignored_token_strings if s.strip()}

    eligible: list[int] = []
    ignored: list[int] = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            ignored.append(token_id)
            continue

        token_text = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if _is_ignorable_surface(
            token_text,
            ignore_punctuation_tokens=ignore_punctuation_tokens,
            ignore_whitespace_tokens=ignore_whitespace_tokens,
            ignored_token_strings=ignored_strings,
        ):
            ignored.append(token_id)
        else:
            eligible.append(token_id)

    return eligible, ignored


def _split_with_plane(
    embeddings: np.ndarray,
    eligible_ids: list[int],
    rng: np.random.Generator,
    max_attempts: int,
) -> tuple[list[int], list[int], PlaneSplitMetrics, np.ndarray, float]:
    """Create split via random plane in embedding space, with boundary tweaking.

    Plane equation is n.x + b = 0, where n is sampled from N(0, I).
    We pick b from the projection median and perform deterministic tie handling
    so we can hit 50:50 (or as close as integer arithmetic allows).
    """

    eligible_emb = embeddings[np.array(eligible_ids, dtype=np.int64)]
    n_tokens = len(eligible_ids)
    target_red = n_tokens // 2

    best = None

    for attempt in range(1, max_attempts + 1):
        normal = rng.standard_normal(size=(eligible_emb.shape[1],), dtype=np.float32)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue
        normal = normal / normal_norm

        projections = eligible_emb @ normal
        sort_idx = np.argsort(projections)
        sorted_proj = projections[sort_idx]

        if n_tokens < 2:
            boundary = float(sorted_proj[0]) if n_tokens else 0.0
        elif n_tokens % 2 == 0:
            lo = sorted_proj[target_red - 1]
            hi = sorted_proj[target_red]
            boundary = float((lo + hi) / 2.0)
        else:
            boundary = float(sorted_proj[target_red])

        # Initial plane assignment.
        red_mask = projections > boundary
        tie_mask = projections == boundary
        tie_count = int(tie_mask.sum())

        red_now = int(red_mask.sum())
        need = target_red - red_now

        if need > 0:
            tie_indices = np.where(tie_mask)[0]
            if len(tie_indices) > 0:
                take = tie_indices[:need]
                red_mask[take] = True
        elif need < 0:
            # Too many reds from > boundary; push lowest-projection reds to blue.
            red_indices = np.where(red_mask)[0]
            if len(red_indices) > target_red:
                red_proj = projections[red_indices]
                order = np.argsort(red_proj)
                move = red_indices[order[: len(red_indices) - target_red]]
                red_mask[move] = False

        # Final deterministic correction in case ties/float edge-cases remain.
        red_indices = np.where(red_mask)[0]
        if len(red_indices) != target_red:
            ordered = np.argsort(projections)
            red_mask[:] = False
            red_mask[ordered[-target_red:]] = True

        red_ids = [eligible_ids[i] for i in np.where(red_mask)[0].tolist()]
        blue_ids = [eligible_ids[i] for i in np.where(~red_mask)[0].tolist()]

        red_ratio = len(red_ids) / max(1, n_tokens)
        blue_ratio = len(blue_ids) / max(1, n_tokens)
        error = abs(red_ratio - 0.5)

        metrics = PlaneSplitMetrics(
            eligible_tokens=n_tokens,
            red_tokens=len(red_ids),
            blue_tokens=len(blue_ids),
            red_ratio=red_ratio,
            blue_ratio=blue_ratio,
            balance_error=error,
            attempts_used=attempt,
            tie_count=tie_count,
        )

        if best is None or metrics.balance_error < best[2].balance_error:
            best = (red_ids, blue_ids, metrics, normal, boundary)

        if metrics.balance_error == 0.0:
            break

    assert best is not None
    return best


def build_plane_split(
    model_name: str,
    output_path: str,
    split_seed: int,
    max_attempts: int,
    ignore_punctuation_tokens: bool,
    ignore_whitespace_tokens: bool,
    ignored_token_strings: list[str],
) -> dict:
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    eligible_ids, ignored_ids = _eligible_token_ids(
        tokenizer,
        ignore_punctuation_tokens=ignore_punctuation_tokens,
        ignore_whitespace_tokens=ignore_whitespace_tokens,
        ignored_token_strings=ignored_token_strings,
    )

    if len(eligible_ids) < 2:
        raise ValueError("Need at least 2 eligible tokens to build a split.")

    rng = np.random.default_rng(split_seed)
    red_ids, blue_ids, metrics, normal, boundary = _split_with_plane(
        embeddings=embeddings,
        eligible_ids=eligible_ids,
        rng=rng,
        max_attempts=max_attempts,
    )

    out = {
        "model_name": model_name,
        "split_method": "embedding_plane",
        "split_seed": split_seed,
        "ignore_punctuation_tokens": ignore_punctuation_tokens,
        "ignore_whitespace_tokens": ignore_whitespace_tokens,
        "ignored_token_strings": ignored_token_strings,
        "plane": {
            "normal": normal.tolist(),
            "bias": -boundary,
            "boundary": boundary,
        },
        "metrics": asdict(metrics),
        "ignored_token_ids": ignored_ids,
        "red_token_ids": sorted(red_ids),
        "blue_token_ids": sorted(blue_ids),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(out, indent=2))

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 50:50 token split from a random plane in embedding space."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--ignore-punctuation-tokens", action="store_true", default=False)
    parser.add_argument("--ignore-whitespace-tokens", action="store_true", default=False)
    parser.add_argument(
        "--ignored-token-strings",
        nargs="*",
        default=["the", "a", "an", "and", "or", "to", "of", "in", "is", "it", "i", "you"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_plane_split(
        model_name=args.model_name,
        output_path=args.output_path,
        split_seed=args.split_seed,
        max_attempts=args.max_attempts,
        ignore_punctuation_tokens=args.ignore_punctuation_tokens,
        ignore_whitespace_tokens=args.ignore_whitespace_tokens,
        ignored_token_strings=args.ignored_token_strings,
    )
    metrics = result["metrics"]
    print(
        "Saved split to {path} | eligible={eligible} red={red} blue={blue} "
        "red_ratio={ratio:.6f} error={error:.6f} attempts={attempts}".format(
            path=args.output_path,
            eligible=metrics["eligible_tokens"],
            red=metrics["red_tokens"],
            blue=metrics["blue_tokens"],
            ratio=metrics["red_ratio"],
            error=metrics["balance_error"],
            attempts=metrics["attempts_used"],
        )
    )


if __name__ == "__main__":
    main()
