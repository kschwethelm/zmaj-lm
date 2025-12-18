#!/usr/bin/env python3
"""Training script for GPT-style language models.

This script loads configuration from a YAML file, initializes model and
dataloaders, and runs the training loop with optional W&B logging.

Example usage:
    python scripts/train.py --config configs/tiny_stories.yaml
    python scripts/train.py --config configs/tiny_stories.yaml --checkpoint-dir checkpoints/run1
"""

import argparse
from pathlib import Path

import torch
import yaml
from loguru import logger

from zmaj_lm.config.dataset_config import DatasetConfig
from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.config.training_config import TrainingConfig
from zmaj_lm.data.dataset import create_dataloaders
from zmaj_lm.models.gpt import GPTModel
from zmaj_lm.training.trainer import Trainer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with parsed YAML configuration
    """
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train a GPT-style language model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints (optional)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    args = parser.parse_args()

    # Load configuration
    config_dict = load_config(args.config)

    # Parse configs using Pydantic models
    model_config = TransformerConfig(**config_dict["model"])
    dataset_config = DatasetConfig(**config_dict["dataset"])
    training_config = TrainingConfig(**config_dict["training"])

    logger.info("Configuration loaded successfully")
    logger.info(
        f"Model: {model_config.num_layers}L x {model_config.num_heads}H x {model_config.hidden_dim}D"
    )
    logger.info(f"Dataset: {dataset_config.dataset_name}")
    logger.info(
        f"Training: {training_config.num_epochs} epochs, lr={training_config.learning_rate}"
    )

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize model
    logger.info("Initializing model...")
    model = GPTModel(config=model_config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")

    # Compile model for performance optimization
    logger.info("Compiling model with torch.compile...")
    model = torch.compile(model)
    logger.info("Model compilation complete")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(dataset_config)
    logger.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume from checkpoint if specified
    if args.resume_from is not None:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Start training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
