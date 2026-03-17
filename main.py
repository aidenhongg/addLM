"""AdditionLM – train an arithmetic CoT model end-to-end."""

import argparse
import re

import torch

from src.model import AdditionLM
from src.tokenization import get_tokenizer
from src.train import load_config, train


def demo(model: AdditionLM, device: torch.device, n: int = 5, enc=None) -> None:
    """Generate a few CoT solutions and print them."""
    import random

    if enc is None:
        enc = get_tokenizer()
    rng = random.Random(0)

    print("\n" + "=" * 50)
    print("Demo: generating CoT solutions")
    print("=" * 50)

    for _ in range(n):
        a = rng.randint(0, 9999)
        b = rng.randint(0, 9999)
        op = rng.choice(["+", "-"])
        expected = (a + b) if op == "+" else (a - b)

        prompt = f"{a} {op} {b}\n"
        prompt_ids = enc.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output_ids = model.generate(idx, max_new_tokens=128, temperature=0.0)
        output_text = enc.decode(output_ids[0].tolist())

        match = re.search(r"=\s*(-?\d+)\s*$", output_text)
        predicted = int(match.group(1)) if match else None
        status = "✓" if predicted == expected else "✗"

        print(f"\n[{status}] {a} {op} {b} = {expected}  (predicted: {predicted})")
        print(output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AdditionLM with CoT")
    parser.add_argument(
        "--config", default="./config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--demo", type=int, default=5, help="Number of demo problems after training (0 to skip)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, enc = train(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.demo and cfg.get("training_stage") == "finetune":
        demo(model, device, n=args.demo, enc=enc)


if __name__ == "__main__":
    main()
