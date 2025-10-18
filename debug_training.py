#!/usr/bin/env python3
"""
Quick training test to debug loss plateau issue.
Run with: python debug_training.py --dataset your_dataset --max-steps 100
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "bin"))
import train
import torch


def debug_training():
    """Run a short training session with enhanced logging to debug plateau."""

    # Use a small batch size and short training for debugging
    parser = train.argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()

    print("Starting debug training session...")
    print(f"Dataset: {args.dataset}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"W&B logging: {args.use_wandb}")

    # Run training with our modifications
    model = train.train_with_preprocessed_dataset(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        use_wandb=args.use_wandb,
        augment=True,  # Enable augmentation
    )

    print("Debug training completed!")


if __name__ == "__main__":
    debug_training()
