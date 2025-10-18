#!/usr/bin/env python3
"""
Test script for optimized TLE training

This script demonstrates the performance improvements from:
1. whisper-large-v2 for Cantonese support
2. Multiprocessing data loading
3. Large batch sizes (no gradient accumulation needed)

Usage:
    python test_optimizations.py --dataset your-dataset --batch-size 32
"""

import argparse
import time
import torch
from bin.train import train_with_preprocessed_dataset


def main():
    parser = argparse.ArgumentParser(description="Test optimized TLE training")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset path or HuggingFace name"
    )
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (large batches for GPU memory)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum training steps for testing"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    print("🚀 Testing Optimized TLE Training")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"Device: {args.device}")
    print()

    print("Expected improvements:")
    print("• whisper-large-v2: Best Cantonese support")
    print("• Multiprocessing: Faster data loading")
    print("• Large batch sizes: Better GPU utilization")
    print("• Overall: Should see >80% GPU utilization")
    print()

    start_time = time.time()

    try:
        # Run optimized training
        model = train_with_preprocessed_dataset(
            dataset_path=args.dataset,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            device=args.device,
            subset=args.subset,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ Training completed successfully!")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Steps per second: {args.max_steps / duration:.2f}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    print("\n💡 Next steps:")
    print("1. Monitor GPU utilization with: watch -n 1 nvidia-smi")
    print("2. Scale up batch size for even better performance")
    print("3. Use --max-epochs instead of --max-steps for full training")

    return 0


if __name__ == "__main__":
    exit(main())
