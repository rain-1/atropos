import hashlib
import json
import math
import random
import string
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)

Color = Literal["red", "blue"]


@dataclass
class RewardBreakdown:
    ratio: float
    ratio_score: float
    direction_score: float
    coherence_score: float
    length_score: float
    final_score: float


class TokenColorBiasConfig(BaseEnvConfig):
    target_ratio: float = Field(
        default=0.70,
        ge=0.50,
        le=0.95,
        description="Desired ratio of target-color tokens among non-ignored completion tokens.",
    )
    ratio_tolerance: float = Field(
        default=0.30,
        ge=0.05,
        le=0.50,
        description="Distance from target ratio where ratio reward linearly falls to zero.",
    )
    split_seed: int = Field(
        default=13,
        description="Seed used to deterministically split vocab IDs into red/blue buckets.",
    )
    token_split_path: Optional[str] = Field(
        default=None,
        description="Optional JSON split file produced by helpers/token_plane_split.py. If set, overrides hash split.",
    )
    ignore_punctuation_tokens: bool = Field(
        default=True,
        description="Ignore tokens that decode to punctuation-only symbols.",
    )
    ignore_whitespace_tokens: bool = Field(
        default=True,
        description="Ignore tokens that decode to pure whitespace.",
    )
    ignored_token_strings: List[str] = Field(
        default_factory=lambda: [
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "is",
            "it",
            "I",
            "you",
        ],
        description="Lower-cased normalized token strings to ignore from color accounting.",
    )
    train_max_tokens: int = Field(default=512)
    eval_max_tokens: int = Field(default=384)
    rollout_temperature: float = Field(default=1.0)
    eval_temperature: float = Field(default=0.2)
    min_colored_tokens: int = Field(
        default=40,
        ge=1,
        description="Minimum number of eligible (non-ignored) completion tokens expected.",
    )
    ratio_reward_weight: float = Field(default=0.50)
    direction_reward_weight: float = Field(default=0.15)
    coherence_reward_weight: float = Field(default=0.25)
    length_reward_weight: float = Field(default=0.10)
    length_finish_penalty: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Penalty subtracted from final reward when finish reason is 'length'.",
    )


class TokenColorBiasEnv(BaseEnv):
    """
    Environment for training controllable lexical bias:
      - model stays coherent/fluently probable (logprob proxy)
      - model follows system instruction to bias red or blue token output
    """

    name = "token_color_bias"
    env_config_cls = TokenColorBiasConfig

    USER_PROMPTS = [
        "Explain one practical way to reduce stress during a busy week.",
        "Write a short response to someone asking for healthy breakfast ideas.",
        "Describe how rain forms in the atmosphere in simple terms.",
        "Give advice for preparing for a technical interview.",
        "Summarize why regular exercise helps long-term health.",
        "Write a helpful reply to someone learning Python for the first time.",
        "Explain how to debug a failing unit test step by step.",
        "Describe two differences between a list and a dictionary in Python.",
        "Provide three tips for writing clear emails at work.",
        "Explain what overfitting means in machine learning.",
    ]

    def __init__(
        self,
        config: TokenColorBiasConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TokenColorBiasConfig = config
        self.iter = 0
        self.train: List[str] = []
        self.test: List[Dict[str, str]] = []
        self.eval_metrics: List[Tuple[str, float]] = []

        self._ignored_token_ids: set[int] = set()
        self._red_token_ids: set[int] = set()
        self._blue_token_ids: set[int] = set()

        self._ratio_buffer: List[float] = []
        self._score_buffer: List[float] = []
        self._coherence_buffer: List[float] = []
        self._target_alignment_buffer: List[float] = []

        self._build_token_split()

    @classmethod
    def config_init(cls) -> Tuple[TokenColorBiasConfig, List[APIServerConfig]]:
        env_config = TokenColorBiasConfig(
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1200,
            batch_size=512,
            steps_per_eval=25,
            max_token_length=2048,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.2,
            wandb_name="token_color_bias",
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=64,
            )
        ]
        return env_config, server_configs

    def _normalize_surface(self, token_text: str) -> str:
        return token_text.replace("▁", " ").strip().lower()

    def _is_ignorable_surface(self, token_text: str) -> bool:
        norm = self._normalize_surface(token_text)
        if self.config.ignore_whitespace_tokens and norm == "":
            return True
        if self.config.ignore_punctuation_tokens and norm and all(
            ch in string.punctuation for ch in norm
        ):
            return True
        if norm in self.config.ignored_token_strings:
            return True
        return False

    def _pick_color(self, token_id: int) -> Color:
        digest = hashlib.sha256(
            f"{self.config.split_seed}:{token_id}".encode("utf-8")
        ).digest()
        return "red" if digest[0] % 2 == 0 else "blue"

    def _build_token_split(self) -> None:
        if self.config.token_split_path:
            self._load_token_split_from_file(self.config.token_split_path)
            return
        self._build_hash_split()

    def _load_token_split_from_file(self, path: str) -> None:
        with open(path, "r") as f:
            split = json.load(f)

        self._red_token_ids = set(split.get("red_token_ids", []))
        self._blue_token_ids = set(split.get("blue_token_ids", []))
        self._ignored_token_ids = set(split.get("ignored_token_ids", []))

        overlap = self._red_token_ids & self._blue_token_ids
        if overlap:
            raise ValueError(
                f"Invalid token split at {path}: red/blue overlap of {len(overlap)} token ids"
            )

        print(
            "Loaded token split from file: "
            f"red={len(self._red_token_ids)}, blue={len(self._blue_token_ids)}, "
            f"ignored={len(self._ignored_token_ids)}"
        )

    def _build_hash_split(self) -> None:
        vocab_size = len(self.tokenizer)
        special_ids = set(self.tokenizer.all_special_ids or [])

        for token_id in range(vocab_size):
            if token_id in special_ids:
                self._ignored_token_ids.add(token_id)
                continue

            token_text = self.tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if self._is_ignorable_surface(token_text):
                self._ignored_token_ids.add(token_id)
                continue

            color = self._pick_color(token_id)
            if color == "red":
                self._red_token_ids.add(token_id)
            else:
                self._blue_token_ids.add(token_id)

        total_colored = len(self._red_token_ids) + len(self._blue_token_ids)
        print(
            f"Token split built from hash: red={len(self._red_token_ids)}, blue={len(self._blue_token_ids)}, "
            f"ignored={len(self._ignored_token_ids)}, colored_total={total_colored}"
        )

    def _score_completion(
        self, target_color: Color, finish_reason: str, masked_tokens: List[int], logprobs: List[float]
    ) -> RewardBreakdown:
        completion_ids = [tok for tok in masked_tokens if tok != -100]

        eligible = 0
        target_count = 0
        for token_id in completion_ids:
            if token_id in self._ignored_token_ids:
                continue
            eligible += 1
            if target_color == "red" and token_id in self._red_token_ids:
                target_count += 1
            elif target_color == "blue" and token_id in self._blue_token_ids:
                target_count += 1

        ratio = target_count / max(eligible, 1)
        ratio_score = max(
            0.0,
            1.0 - abs(ratio - self.config.target_ratio) / self.config.ratio_tolerance
        )

        direction_score = 1.0 if ratio >= 0.5 else 0.0

        completion_logprobs = [
            lp for tok, lp in zip(masked_tokens, logprobs, strict=False) if tok != -100
        ]
        mean_lp = sum(completion_logprobs) / max(len(completion_logprobs), 1)
        coherence_score = 1.0 / (1.0 + math.exp(-(mean_lp + 2.0) / 0.7))

        length_score = min(1.0, eligible / self.config.min_colored_tokens)

        final_score = (
            self.config.ratio_reward_weight * ratio_score
            + self.config.direction_reward_weight * direction_score
            + self.config.coherence_reward_weight * coherence_score
            + self.config.length_reward_weight * length_score
        )

        if finish_reason == "length":
            final_score -= self.config.length_finish_penalty

        final_score = max(0.0, min(1.0, final_score))

        return RewardBreakdown(
            ratio=ratio,
            ratio_score=ratio_score,
            direction_score=direction_score,
            coherence_score=coherence_score,
            length_score=length_score,
            final_score=final_score,
        )

    async def setup(self):
        prompts = list(self.USER_PROMPTS)
        random.Random(42).shuffle(prompts)

        split_idx = max(1, int(len(prompts) * 0.8))
        self.train = prompts[:split_idx]

        self.test = []
        for prompt in prompts[split_idx:]:
            self.test.append({"prompt": prompt, "target_color": "red"})
            self.test.append({"prompt": prompt, "target_color": "blue"})

        self.iter = 0

    async def get_next_item(self) -> Item:
        prompt = self.train[self.iter % len(self.train)]
        target_color: Color = "red" if random.random() < 0.5 else "blue"
        self.iter += 1
        return {
            "prompt": prompt,
            "target_color": target_color,
            "target_ratio": self.config.target_ratio,
        }

    def _system_prompt_for_target(self, target_color: Color, target_ratio: float) -> str:
        return (
            "You are a helpful, coherent assistant. Keep your response fluent and meaningful. "
            f"When possible, bias your output toward {target_color.upper()} tokens. "
            f"Aim for about {int(target_ratio * 100)}% {target_color.upper()} tokens "
            "among lexical tokens while still answering naturally and accurately."
        )

    async def collect_trajectories(self, item) -> Tuple[Optional[ScoredDataGroup], List]:
        target_color: Color = item["target_color"]
        target_ratio = float(item["target_ratio"])

        messages = [
            {
                "role": "system",
                "content": self._system_prompt_for_target(target_color, target_ratio),
            },
            {"role": "user", "content": item["prompt"]},
        ]

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=self.config.train_max_tokens,
                temperature=self.config.rollout_temperature,
            )
            nodes = managed.get_state()["nodes"]

        scored = ScoredDataGroup(tokens=[], masks=[], scores=[], inference_logprobs=[])

        for choice, node in zip(completion.choices, nodes, strict=False):
            breakdown = self._score_completion(
                target_color=target_color,
                finish_reason=choice.finish_reason,
                masked_tokens=node.masked_tokens,
                logprobs=node.logprobs,
            )
            scored["tokens"].append(node.tokens)
            scored["masks"].append(node.masked_tokens)
            scored["inference_logprobs"].append(node.logprobs)
            scored["scores"].append(breakdown.final_score)

            self._ratio_buffer.append(breakdown.ratio)
            self._score_buffer.append(breakdown.final_score)
            self._coherence_buffer.append(breakdown.coherence_score)
            self._target_alignment_buffer.append(breakdown.ratio_score)

        if len(set(scored["scores"])) <= 1 and self.config.ensure_scores_are_not_same:
            return None, []

        return scored, []

    async def evaluate(self, *args, **kwargs):
        if not self.test:
            return

        sample_count = min(32, len(self.test))
        eval_items = random.sample(self.test, sample_count)

        async def eval_one(item_eval):
            messages = [
                {
                    "role": "system",
                    "content": self._system_prompt_for_target(
                        item_eval["target_color"], self.config.target_ratio
                    ),
                },
                {"role": "user", "content": item_eval["prompt"]},
            ]
            async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                completion = await managed.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=self.config.eval_max_tokens,
                    temperature=self.config.eval_temperature,
                )
                node = managed.get_state()["nodes"][0]

            breakdown = self._score_completion(
                target_color=item_eval["target_color"],
                finish_reason=completion.choices[0].finish_reason,
                masked_tokens=node.masked_tokens,
                logprobs=node.logprobs,
            )
            return breakdown

        results = await tqdm_asyncio.gather(*[eval_one(x) for x in eval_items])
        avg_score = sum(x.final_score for x in results) / len(results)
        avg_ratio_err = sum(abs(x.ratio - self.config.target_ratio) for x in results) / len(
            results
        )
        avg_coherence = sum(x.coherence_score for x in results) / len(results)

        self.eval_metrics = [
            ("eval/avg_score", avg_score),
            ("eval/avg_ratio_error", avg_ratio_err),
            ("eval/avg_coherence", avg_coherence),
        ]

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self._score_buffer:
            wandb_metrics["train/avg_score"] = sum(self._score_buffer) / len(
                self._score_buffer
            )
            wandb_metrics["train/avg_target_ratio"] = sum(self._ratio_buffer) / len(
                self._ratio_buffer
            )
            wandb_metrics["train/avg_ratio_alignment"] = sum(
                self._target_alignment_buffer
            ) / len(self._target_alignment_buffer)
            wandb_metrics["train/avg_coherence"] = sum(self._coherence_buffer) / len(
                self._coherence_buffer
            )

        for key, value in self.eval_metrics:
            wandb_metrics[key] = value

        self._score_buffer = []
        self._ratio_buffer = []
        self._coherence_buffer = []
        self._target_alignment_buffer = []
        self.eval_metrics = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    TokenColorBiasEnv.cli()
