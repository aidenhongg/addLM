"""Chain-of-Thought data generation and dataset for arithmetic."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.model import IGNORE_INDEX
from src.tokenization import get_tokenizer


# ── CoT formatting ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoTExample:
    """A single arithmetic problem with step-by-step reasoning."""

    prompt: str
    reasoning: str
    answer: str

    @property
    def full_text(self) -> str:
        return self.prompt + self.reasoning + self.answer


class CoTFormatter:
    """Generates digit-by-digit reasoning traces for addition and subtraction.

    Example output for ``123 + 456``::

        123 + 456
        3+6+0=9 c0
        2+5+0=7 c0
        1+4+0=5 c0
        = 579
    """

    @staticmethod
    def format(a: int, b: int, op: str) -> CoTExample:
        if a < 0 or b < 0:
            raise ValueError("Operands must be non-negative")
        if op == "+":
            return CoTFormatter._addition(a, b)
        if op == "-":
            return CoTFormatter._subtraction(a, b)
        raise ValueError(f"Unsupported operator: {op!r}")

    # ── public helpers for direct use ────────────────────────────────────

    @staticmethod
    def _addition(a: int, b: int) -> CoTExample:
        result = a + b
        steps = CoTFormatter._add_digit_steps(a, b)
        return CoTExample(
            prompt=f"{a} + {b}\n",
            reasoning="\n".join(steps) + "\n",
            answer=f"= {result}",
        )

    @staticmethod
    def _subtraction(a: int, b: int) -> CoTExample:
        result = a - b
        if a >= b:
            steps = CoTFormatter._sub_digit_steps(a, b)
        else:
            # |a| < |b|  →  negate the unsigned result
            steps = ["NEG"] + CoTFormatter._sub_digit_steps(b, a)
        return CoTExample(
            prompt=f"{a} - {b}\n",
            reasoning="\n".join(steps) + "\n",
            answer=f"= {result}",
        )

    # ── digit-level step generators ──────────────────────────────────────

    @staticmethod
    def _add_digit_steps(a: int, b: int) -> list[str]:
        """Right-to-left column addition with carry tracking."""
        sa, sb = str(a), str(b)
        width = max(len(sa), len(sb))
        sa, sb = sa.zfill(width), sb.zfill(width)

        carry = 0
        steps: list[str] = []
        for i in range(width - 1, -1, -1):
            da, db = int(sa[i]), int(sb[i])
            total = da + db + carry
            digit = total % 10
            new_carry = total // 10
            steps.append(f"{da}+{db}+{carry}={digit} c{new_carry}")
            carry = new_carry
        if carry:
            steps.append(f"c{carry}")
        return steps

    @staticmethod
    def _sub_digit_steps(a: int, b: int) -> list[str]:
        """Right-to-left column subtraction with borrow tracking.  Requires a >= b."""
        sa, sb = str(a), str(b)
        width = len(sa)
        sb = sb.zfill(width)

        borrow = 0
        steps: list[str] = []
        for i in range(width - 1, -1, -1):
            da, db = int(sa[i]), int(sb[i])
            diff = da - db - borrow
            new_borrow = 0
            if diff < 0:
                diff += 10
                new_borrow = 1
            steps.append(f"{da}-{db}-{borrow}={diff} b{new_borrow}")
            borrow = new_borrow
        return steps


# ── Text collection for tokenizer training ───────────────────────────────────


def collect_texts(
    datasets: dict, max_texts: int = 50_000, seed: int = 42
) -> list[str]:
    """Extract raw text from all data sources for tokenizer vocabulary building."""
    texts: list[str] = []
    for a, b, op in datasets["math_equations"]:
        ex = CoTFormatter.format(a, b, op)
        texts.append(ex.full_text)
    for split in datasets["math_stories"].values():
        for row in split:
            texts.append(row["story_1_qs"] + "\n" + str(row["answer"]))
    for split in datasets["tiny_stories"].values():
        for row in split:
            texts.append(row["text"])
    if len(texts) > max_texts:
        rng = random.Random(seed)
        texts = rng.sample(texts, max_texts)
    return texts


# ── Dataset ──────────────────────────────────────────────────────────────────


class MathCoTDataset(Dataset):
    """Unified dataset aggregating multiple sources into tokenised (prompt, answer) pairs.

    Sources:
    - ``math_equations``: ``(a, b, op)`` tuples → CoT-formatted with reasoning traces
    - ``math_stories``: HF dataset with question fields → prompt / answer pairs
    - ``tiny_stories``: HF dataset with text → split in half as prompt / answer

    Each item returns ``(input_ids, target_ids)`` where prompt tokens in
    ``target_ids`` are set to ``IGNORE_INDEX`` so the loss only covers the
    answer portion.
    """

    def __init__(self, datasets: dict, max_seq_len: int = 512, seed: int = 42, enc=None):
        if enc is None:
            enc = get_tokenizer()
        rng = random.Random(seed)

        self.inputs: list[Tensor] = []
        self.targets: list[Tensor] = []

        for a, b, op in datasets["math_equations"]:
            ex = CoTFormatter.format(a, b, op)
            self._add(enc, ex.prompt, ex.reasoning + ex.answer, max_seq_len)

        for split in datasets["math_stories"].values():
            for row in split:
                answer = str(row["answer"])
                self._add(enc, row["story_1_qs"] + "\n", answer, max_seq_len)

        for split in datasets["tiny_stories"].values():
            for row in split:
                text = row["text"]
                mid = len(text) // 2
                if mid > 0:
                    self._add(enc, text[:mid], text[mid:], max_seq_len)

        indices = list(range(len(self.inputs)))
        rng.shuffle(indices)
        self.inputs = [self.inputs[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]

    def _add(self, enc, prompt: str, answer: str, max_seq_len: int) -> None:
        full_tokens = enc.encode(prompt + answer)
        if len(full_tokens) > max_seq_len:
            return
        prompt_len = len(enc.encode(prompt))
        tokens = torch.tensor(full_tokens, dtype=torch.long)
        inp = tokens[:-1]
        tgt = tokens[1:].clone()
        if prompt_len > 1:
            tgt[: prompt_len - 1] = IGNORE_INDEX
        self.inputs.append(inp)
        self.targets.append(tgt)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.inputs[idx], self.targets[idx]


def collate_cot(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Pad variable-length CoT examples into a uniform batch.

    Padding positions in ``targets`` are set to ``IGNORE_INDEX`` so they are
    ignored by the loss.
    """
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.full(
        (len(targets), max_len), IGNORE_INDEX, dtype=torch.long
    )

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        length = inp.size(0)
        padded_inputs[i, :length] = inp
        padded_targets[i, :length] = tgt

    return padded_inputs, padded_targets
