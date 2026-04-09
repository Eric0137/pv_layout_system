"""
Model Training Script
=====================
Standalone script to train the obstacle classifier CNN.
Can use synthetic data for bootstrapping or real labeled images.

Usage:
    python train_model.py --mode synthetic --samples 300
    python train_model.py --mode real --dataset ./dataset/obstacles
    python train_model.py --mode both --samples 200
"""

import os
import sys
import argparse
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OBSTACLE_DATASET_DIR, MODEL_DIR, CNN_CONFIG
from obstacle_classifier import (
    ObstacleClassifier,
    SyntheticObstacleGenerator,
    ObstacleDataset,
    TORCH_AVAILABLE
)
from visualization import ChartGenerator


def train_with_synthetic_data(samples_per_class: int = 200):
    """Generate synthetic data and train the model."""
    print("=" * 60)
    print("OBSTACLE CLASSIFIER TRAINING (Synthetic Data)")
    print("=" * 60)

    # Step 1: Generate synthetic dataset
    print("\n[STEP 1] Generating synthetic training images...")
    generator = SyntheticObstacleGenerator()
    generator.generate_dataset(
        output_dir=OBSTACLE_DATASET_DIR,
        samples_per_class=samples_per_class
    )

    # Step 2: Train model
    return _train_model()


def train_with_real_data(dataset_dir: str = None):
    """Train using real labeled images."""
    print("=" * 60)
    print("OBSTACLE CLASSIFIER TRAINING (Real Data)")
    print("=" * 60)

    dataset_dir = dataset_dir or OBSTACLE_DATASET_DIR
    dataset = ObstacleDataset(dataset_dir)
    dist = dataset.get_class_distribution()
    print(f"\n[INFO] Dataset directory: {dataset_dir}")
    print(f"[INFO] Class distribution: {dist}")

    total = sum(dist.values())
    if total < 20:
        print(f"[WARNING] Only {total} images found. Minimum ~50 recommended.")
        print("[TIP] Add more images to class subfolders or use synthetic mode.")
        if total == 0:
            print("[ERROR] No images found. Cannot train.")
            return None

    return _train_model(dataset_dir)


def train_with_both(samples_per_class: int = 200, dataset_dir: str = None):
    """Combine synthetic and real data for training."""
    print("=" * 60)
    print("OBSTACLE CLASSIFIER TRAINING (Synthetic + Real Data)")
    print("=" * 60)

    dataset_dir = dataset_dir or OBSTACLE_DATASET_DIR

    # Generate synthetic data (supplements existing real data)
    print("\n[STEP 1] Generating synthetic training images...")
    generator = SyntheticObstacleGenerator()
    generator.generate_dataset(
        output_dir=dataset_dir,
        samples_per_class=samples_per_class
    )

    return _train_model(dataset_dir)


def _train_model(dataset_dir: str = None):
    """Common training logic."""
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required. Install: pip install torch torchvision")
        return None

    dataset_dir = dataset_dir or OBSTACLE_DATASET_DIR

    # Initialize classifier
    classifier = ObstacleClassifier()

    # Step: Prepare data
    print("\n[STEP 2] Preparing datasets...")
    train_loader, val_loader, test_loader = classifier.prepare_datasets(dataset_dir)

    # Step: Train
    print("\n[STEP 3] Training CNN...")
    history = classifier.train(
        train_loader, val_loader,
        epochs=CNN_CONFIG["epochs"],
        lr=CNN_CONFIG["learning_rate"]
    )

    # Step: Evaluate
    print("\n[STEP 4] Evaluating on test set...")
    metrics = classifier.evaluate(test_loader)
    print(f"Test Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Per-class: {metrics['class_accuracy']}")

    # Step: Save training charts
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
    chart_path = os.path.join(
        os.path.dirname(__file__), "outputs", "training_curves.png"
    )
    ChartGenerator.training_history_chart(history, save_path=chart_path)
    print(f"\n[SAVED] Training curves: {chart_path}")

    # Save metrics
    metrics_path = os.path.join(
        os.path.dirname(__file__), "outputs", "training_metrics.json"
    )
    with open(metrics_path, 'w') as f:
        json.dump({
            "test_accuracy": metrics["overall_accuracy"],
            "class_accuracy": metrics["class_accuracy"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "epochs_trained": len(history["train_loss"]),
        }, f, indent=2)
    print(f"[SAVED] Metrics: {metrics_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {CNN_CONFIG['model_save_path']}")
    print("=" * 60)

    return history, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train the PV Layout Obstacle Classifier"
    )
    parser.add_argument(
        "--mode", type=str, default="synthetic",
        choices=["synthetic", "real", "both"],
        help="Training mode: synthetic (generated), real (your images), both"
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Number of synthetic samples per class (default: 200)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to dataset directory (default: ./dataset/obstacles)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )

    args = parser.parse_args()

    # Override config if specified
    if args.epochs:
        CNN_CONFIG["epochs"] = args.epochs
    if args.lr:
        CNN_CONFIG["learning_rate"] = args.lr

    if args.mode == "synthetic":
        train_with_synthetic_data(args.samples)
    elif args.mode == "real":
        train_with_real_data(args.dataset)
    elif args.mode == "both":
        train_with_both(args.samples, args.dataset)


if __name__ == "__main__":
    main()
