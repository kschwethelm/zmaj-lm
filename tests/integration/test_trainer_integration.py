"""Integration tests for Trainer with full training loop."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.config.training_config import TrainingConfig
from zmaj_lm.models.gpt import GPTModel
from zmaj_lm.training.trainer import Trainer


@pytest.fixture
def small_model(device: torch.device) -> nn.Module:
    """Create a small GPT model for testing."""
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout_rate=0.0,
    )
    return GPTModel(config=config).to(device)


@pytest.fixture
def train_dataloader() -> DataLoader:
    """Create a small training dataloader."""
    batch_size = 4
    seq_len = 16
    vocab_size = 100
    num_samples = 16

    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    dataset = TensorDataset(input_ids, target_ids)

    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor | None]:
        input_ids_list, target_ids_list = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids_list),
            "target_ids": torch.stack(target_ids_list),
            "attention_mask": None,  # No masking for simple tests
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


@pytest.fixture
def val_dataloader() -> DataLoader:
    """Create a small validation dataloader."""
    batch_size = 4
    seq_len = 16
    vocab_size = 100
    num_samples = 8

    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    dataset = TensorDataset(input_ids, target_ids)

    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor | None]:
        input_ids_list, target_ids_list = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids_list),
            "target_ids": torch.stack(target_ids_list),
            "attention_mask": None,  # No masking for simple tests
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


@pytest.mark.integration
class TestTrainerIntegration:
    """Integration tests for complete training loop."""

    def test_full_training_loop(
        self,
        small_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Test that full training loop completes successfully."""
        config = TrainingConfig(
            seed=42,
            learning_rate=1e-3,
            batch_size=4,
            num_epochs=2,
            warmup_steps=2,
            eval_every_n_steps=4,
            checkpoint_every_n_steps=8,
            log_every_n_steps=2,
            scheduler_type="cosine",
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        # Run training
        trainer.train()

        # Check that training completed
        assert trainer.global_step > 0
        assert trainer.current_epoch >= 0

        # Check that final loss is finite
        batch = next(iter(train_dataloader))
        final_metrics = trainer.train_step(batch)
        final_loss = final_metrics["loss"]

        # Note: With random data, loss might not always decrease, but should be finite
        assert torch.isfinite(torch.tensor(final_loss))

    def test_training_with_max_steps(
        self,
        small_model: nn.Module,
        train_dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Test training with max_steps limit."""
        config = TrainingConfig(
            seed=42,
            learning_rate=1e-3,
            num_epochs=100,  # Would be very long
            max_steps=10,  # But stop after 10 steps
            warmup_steps=2,
            log_every_n_steps=2,
            scheduler_type="cosine",
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        trainer.train()

        assert trainer.global_step == 10

    def test_checkpoint_saving_during_training(
        self,
        small_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Test that checkpoints are saved during training."""
        config = TrainingConfig(
            seed=42,
            learning_rate=1e-3,
            num_epochs=1,
            warmup_steps=1,
            eval_every_n_steps=2,
            checkpoint_every_n_steps=4,
            log_every_n_steps=1,
            scheduler_type="cosine",
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer.train()

        # Check that checkpoints were saved
        checkpoints = list(tmp_path.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

    def test_best_model_saving(
        self,
        small_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Test that best model is saved during training."""
        config = TrainingConfig(
            seed=42,
            learning_rate=1e-3,
            num_epochs=1,
            warmup_steps=1,
            eval_every_n_steps=2,
            checkpoint_every_n_steps=10,
            log_every_n_steps=1,
            scheduler_type="cosine",
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer.train()

        # Check that best model was saved
        best_model_path = tmp_path / "best_model.pt"
        assert best_model_path.exists()

    def test_learning_rate_schedule(
        self,
        small_model: nn.Module,
        train_dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Test that learning rate follows the schedule."""
        config = TrainingConfig(
            seed=42,
            learning_rate=1e-3,
            num_epochs=1,
            warmup_steps=3,
            log_every_n_steps=1,
            scheduler_type="cosine",
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        learning_rates = []

        # Record LR during warmup and beyond
        for _ in range(6):
            batch = next(iter(train_dataloader))
            trainer.train_step(batch)
            learning_rates.append(trainer.scheduler.get_last_lr()[0])

        # During warmup, LR should increase
        assert learning_rates[1] > learning_rates[0]
        assert learning_rates[2] > learning_rates[1]

        # After warmup, with cosine schedule, LR should eventually decrease
        assert learning_rates[-1] < max(learning_rates)

    def test_training_with_constant_scheduler(
        self,
        small_model: nn.Module,
        train_dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Test training with constant_with_warmup scheduler."""
        config = TrainingConfig(
            seed=42,
            learning_rate=1e-3,
            num_epochs=1,
            warmup_steps=2,
            log_every_n_steps=1,
            scheduler_type="constant",
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        trainer.train()

        # Should complete without errors
        assert trainer.global_step > 0

    @pytest.mark.slow
    def test_resume_from_checkpoint(
        self,
        small_model: nn.Module,
        train_dataloader: DataLoader,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Test resuming training from a checkpoint."""
        config = TrainingConfig(
            seed=42,
            learning_rate=1e-3,
            num_epochs=1,
            max_steps=5,
            warmup_steps=1,
            checkpoint_every_n_steps=3,
            log_every_n_steps=1,
        )

        # Train for a few steps
        trainer1 = Trainer(
            model=small_model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer1.train()

        # Find the checkpoint
        checkpoints = list(tmp_path.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0
        checkpoint_path = checkpoints[0]

        # Resume from checkpoint
        new_model = GPTModel(
            config=TransformerConfig(
                vocab_size=100,
                max_seq_len=32,
                hidden_dim=64,
                num_layers=2,
                num_heads=2,
                dropout_rate=0.0,
            )
        ).to(device)

        trainer2 = Trainer(
            model=new_model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer2.load_checkpoint(checkpoint_path)

        # Check that state was restored from the checkpoint (not final trainer1 state)
        # Extract step number from checkpoint filename (e.g., "checkpoint_step_3.pt" -> 3)
        checkpoint_step = int(checkpoint_path.stem.split("_")[-1])
        assert trainer2.global_step == checkpoint_step
        assert trainer2.current_epoch == 0  # Should be epoch 0 since checkpoint was saved early
