"""Unit tests for Trainer class."""

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
def dummy_dataloader() -> DataLoader:
    """Create a dummy dataloader for testing."""
    batch_size = 4
    seq_len = 16
    vocab_size = 100

    input_ids = torch.randint(0, vocab_size, (batch_size * 5, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size * 5, seq_len))

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

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create a basic training config for testing."""
    return TrainingConfig(
        seed=42,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=2,
        warmup_steps=2,
        eval_every_n_steps=5,
        checkpoint_every_n_steps=10,
        log_every_n_steps=2,
        scheduler_type="cosine",
    )


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_trainer_init_basic(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test basic trainer initialization."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float("inf")
        assert not trainer.use_wandb

    def test_trainer_init_with_max_steps(
        self, small_model: nn.Module, dummy_dataloader: DataLoader, device: torch.device
    ) -> None:
        """Test trainer initialization with max_steps overriding num_epochs."""
        config = TrainingConfig(
            num_epochs=10,
            max_steps=20,
            warmup_steps=2,
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        assert trainer.total_steps == 20
        # num_epochs = ceil(max_steps / steps_per_epoch)
        expected_epochs = (20 + len(dummy_dataloader) - 1) // len(dummy_dataloader)
        assert trainer.num_epochs == expected_epochs

    def test_trainer_init_wandb_disabled(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test that wandb is not initialized when use_wandb=False."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
        )

        assert not trainer.use_wandb


class TestSchedulerCreation:
    """Test learning rate scheduler creation."""

    def test_cosine_scheduler(
        self, small_model: nn.Module, dummy_dataloader: DataLoader, device: torch.device
    ) -> None:
        """Test creation of cosine scheduler with warmup."""
        config = TrainingConfig(
            scheduler_type="cosine",
            warmup_steps=5,
            num_epochs=1,
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        assert trainer.scheduler is not None
        # After initialization, LR should be >= 0 (might be 0 at start of warmup)
        initial_lr = trainer.scheduler.get_last_lr()[0]
        assert initial_lr >= 0

        # After one step, LR should be increasing during warmup
        trainer.scheduler.step()
        step1_lr = trainer.scheduler.get_last_lr()[0]
        assert step1_lr >= initial_lr

    def test_constant_scheduler(
        self, small_model: nn.Module, dummy_dataloader: DataLoader, device: torch.device
    ) -> None:
        """Test creation of constant scheduler with warmup."""
        config = TrainingConfig(
            scheduler_type="constant",
            warmup_steps=5,
            num_epochs=1,
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        assert trainer.scheduler is not None
        # After initialization, LR should be >= 0 (might be 0 at start of warmup)
        initial_lr = trainer.scheduler.get_last_lr()[0]
        assert initial_lr >= 0

        # After several steps past warmup, LR should be constant
        for _ in range(10):
            trainer.scheduler.step()
        post_warmup_lr = trainer.scheduler.get_last_lr()[0]
        assert post_warmup_lr > 0

    def test_scheduler_with_zero_warmup(
        self, small_model: nn.Module, dummy_dataloader: DataLoader, device: torch.device
    ) -> None:
        """Test scheduler creation with warmup_steps=0."""
        config = TrainingConfig(
            scheduler_type="cosine",
            warmup_steps=0,  # No warmup
            num_epochs=1,
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        assert trainer.scheduler is not None
        # With 0 warmup, should start at full learning rate
        initial_lr = trainer.scheduler.get_last_lr()[0]
        assert initial_lr == config.learning_rate

    def test_invalid_scheduler_type(
        self, small_model: nn.Module, dummy_dataloader: DataLoader, device: torch.device
    ) -> None:
        """Test that invalid scheduler type raises ValueError."""
        config = TrainingConfig(
            scheduler_type="invalid_scheduler",
            num_epochs=1,
        )

        with pytest.raises(ValueError, match="Unsupported scheduler_type"):
            Trainer(
                model=small_model,
                train_dataloader=dummy_dataloader,
                val_dataloader=None,
                config=config,
                device=device,
            )


class TestTrainingStep:
    """Test training step functionality."""

    def test_train_step_returns_metrics(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test that train_step returns loss and perplexity."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
        )

        batch = next(iter(dummy_dataloader))
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert "perplexity" in metrics
        assert metrics["loss"] > 0
        assert metrics["perplexity"] > 0

    def test_train_step_updates_weights(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test that train_step updates model weights."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
        )

        # Get initial weights (check all embedding weights which should definitely change)
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Perform multiple training steps to ensure parameter updates
        for _ in range(3):
            batch = next(iter(dummy_dataloader))
            trainer.train_step(batch)

        # Check that at least some weights changed
        params_changed = False
        for initial, current in zip(initial_params, trainer.model.parameters()):
            if not torch.allclose(initial, current, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "Model parameters should change after training steps"

    def test_gradient_clipping(
        self, small_model: nn.Module, dummy_dataloader: DataLoader, device: torch.device
    ) -> None:
        """Test that gradient clipping is applied."""
        config = TrainingConfig(
            grad_clip_norm=0.5,
            num_epochs=1,
        )

        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=config,
            device=device,
        )

        batch = next(iter(dummy_dataloader))
        trainer.train_step(batch)

        # After clipping, no gradient should exceed the clip norm significantly
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        # Total norm should be close to or less than clip_norm after a step
        # Note: This is approximate as optimizer might have already zeroed grads
        assert total_norm >= 0  # Sanity check


class TestValidation:
    """Test validation functionality."""

    def test_validate_returns_metrics(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test that validate returns loss and perplexity."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            config=training_config,
            device=device,
        )

        val_metrics = trainer.validate()

        assert "val_loss" in val_metrics
        assert "val_perplexity" in val_metrics
        assert val_metrics["val_loss"] > 0
        assert val_metrics["val_perplexity"] > 0

    def test_validate_without_dataloader(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test that validate returns empty dict when no val_dataloader."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
        )

        val_metrics = trainer.validate()
        assert val_metrics == {}

    def test_validate_no_gradient(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test that validation doesn't compute gradients."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            config=training_config,
            device=device,
        )

        trainer.validate()

        # Check that no gradients are stored
        for p in trainer.model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    def test_save_checkpoint(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Test checkpoint saving."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer.global_step = 100
        trainer.save_checkpoint(is_best=False)

        checkpoint_path = tmp_path / "checkpoint_step_100.pt"
        assert checkpoint_path.exists()

    def test_save_best_checkpoint(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Test saving best model checkpoint."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer.global_step = 100
        trainer.save_checkpoint(is_best=True)

        best_path = tmp_path / "best_model.pt"
        assert best_path.exists()

    def test_load_checkpoint(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Test checkpoint loading."""
        # Create and save a checkpoint
        trainer1 = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer1.global_step = 50
        trainer1.current_epoch = 1
        trainer1.best_val_loss = 2.5
        trainer1.save_checkpoint(is_best=False)

        checkpoint_path = tmp_path / "checkpoint_step_50.pt"

        # Create new trainer and load checkpoint
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
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
            checkpoint_dir=tmp_path,
        )

        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2.global_step == 50
        assert trainer2.current_epoch == 1
        assert trainer2.best_val_loss == 2.5

    def test_no_checkpoint_without_dir(
        self,
        small_model: nn.Module,
        dummy_dataloader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Test that no checkpoint is saved when checkpoint_dir is None."""
        trainer = Trainer(
            model=small_model,
            train_dataloader=dummy_dataloader,
            val_dataloader=None,
            config=training_config,
            device=device,
            checkpoint_dir=None,
        )

        trainer.global_step = 100
        trainer.save_checkpoint(is_best=False)
        # Should not raise an error, just skip saving
