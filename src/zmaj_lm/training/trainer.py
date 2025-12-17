from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from zmaj_lm.config.training_config import TrainingConfig

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Training loop for language model pre-training.

    Handles:
    - Forward/backward passes with gradient clipping
    - Learning rate scheduling (cosine with warmup)
    - Training and validation loops
    - Metrics tracking (loss, perplexity)
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Language model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on (CPU/CUDA)
            checkpoint_dir: Directory to save checkpoints (optional)
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Initialize W&B
        if config.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb requested but not installed. Install with: uv add wandb")
                self.use_wandb = False
            else:
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.model_dump(),
                )
                wandb.watch(model, log="all", log_freq=config.log_every_n_steps)
                self.use_wandb = True
        else:
            self.use_wandb = False

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        # Calculate total training steps
        steps_per_epoch = len(train_dataloader)
        if config.max_steps is not None:
            self.total_steps = config.max_steps
            self.num_epochs = (config.max_steps + steps_per_epoch - 1) // steps_per_epoch
        else:
            self.total_steps = steps_per_epoch * config.num_epochs
            self.num_epochs = config.num_epochs

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Initialize SWA/EMA if requested
        self.averaged_model: AveragedModel | None = None
        if config.use_swa:
            # Determine decay function: EMA if swa_decay is set, otherwise equal averaging (SWA)
            if config.swa_decay is not None:
                # EMA with decay
                def ema_avg_fn(
                    averaged_param: torch.Tensor,
                    model_param: torch.Tensor,
                    _num_averaged: torch.Tensor | int,
                ) -> torch.Tensor:
                    return config.swa_decay * averaged_param + (1 - config.swa_decay) * model_param

                avg_fn = ema_avg_fn
                logger.info(f"Initializing EMA with decay={config.swa_decay}")
            else:
                # Equal averaging (SWA)
                avg_fn = None
                logger.info("Initializing SWA with equal averaging")

            self.averaged_model = AveragedModel(model, avg_fn=avg_fn)

            # Determine when to start averaging
            self.swa_start_step = (
                config.swa_start_step if config.swa_start_step is not None else config.warmup_steps
            )
            logger.info(f"Weight averaging will start at step {self.swa_start_step}")

        # Initialize tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info(
            f"Trainer initialized with {self.total_steps} total steps over {self.num_epochs} epochs"
        )
        logger.info(f"Training on device: {device}")
        if self.use_wandb:
            logger.info(
                f"W&B logging enabled: project={config.wandb_project}, run={config.wandb_run_name}"
            )

    def _create_scheduler(self) -> LRScheduler:
        """Create learning rate scheduler based on config.

        Returns:
            Learning rate scheduler

        Raises:
            ValueError: If scheduler_type is not supported
        """
        scheduler_type = self.config.scheduler_type.lower()

        if scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif scheduler_type == "constant":
            return get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
            )
        else:
            raise ValueError(
                f"Unsupported scheduler_type: {self.config.scheduler_type}. "
                f"Supported types: 'cosine', 'constant'"
            )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute single training step.

        Args:
            batch: Batch containing input_ids, target_ids, attention_mask

        Returns:
            Dictionary with 'loss' and 'perplexity'
        """
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass
        logits = self.model(input_ids, attention_mask=attention_mask)

        # Compute loss
        # Reshape for cross-entropy: (batch * seq_len, vocab_size) and (batch * seq_len)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Compute metrics
        perplexity = torch.exp(loss).item()

        return {
            "loss": loss.item(),
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation loop.

        Uses the averaged model if SWA/EMA is enabled and swa_eval is True.

        Returns:
            Dictionary with 'val_loss' and 'val_perplexity'
        """
        if self.val_dataloader is None:
            return {}

        # Select model for evaluation
        eval_model = self._get_eval_model()
        eval_model.eval()

        total_loss = 0.0
        num_batches = 0

        val_pbar = tqdm(self.val_dataloader, desc="Validation", leave=False)
        for batch in val_pbar:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Forward pass
            logits = eval_model(input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current average metrics
            avg_loss = total_loss / num_batches
            val_pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "ppl": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                }
            )

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }

    def _get_eval_model(self) -> nn.Module:
        """Get the model to use for evaluation.

        Returns:
            Averaged model if SWA/EMA is enabled and swa_eval is True,
            otherwise the regular model.
        """
        if (
            self.averaged_model is not None
            and self.config.swa_eval
            and self.global_step >= self.swa_start_step
        ):
            return self.averaged_model.module
        return self.model

    def train(self) -> None:
        """Execute full training loop."""
        logger.info("Starting training...")

        epoch_pbar = tqdm(range(self.num_epochs), desc="Epochs", position=0)
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            batch_pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                position=1,
                leave=False,
            )
            for batch in batch_pbar:
                # Training step
                metrics = self.train_step(batch)
                self.global_step += 1

                # Update averaged model if SWA/EMA is enabled and we're past the start step
                if self.averaged_model is not None and self.global_step >= self.swa_start_step:
                    self.averaged_model.update_parameters(self.model)

                # Update progress bar with live metrics
                lr = self.scheduler.get_last_lr()[0]
                batch_pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "ppl": f"{metrics['perplexity']:.2f}",
                        "lr": f"{lr:.2e}",
                    }
                )

                # Log training metrics
                if self.global_step % self.config.log_every_n_steps == 0:
                    logger.info(
                        f"Step {self.global_step}/{self.total_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Perplexity: {metrics['perplexity']:.2f} | "
                        f"LR: {lr:.2e}"
                    )

                    if self.use_wandb:
                        wandb.log(
                            {
                                "train/loss": metrics["loss"],
                                "train/perplexity": metrics["perplexity"],
                                "train/learning_rate": lr,
                                "train/epoch": self.current_epoch,
                            },
                            step=self.global_step,
                        )

                # Validation
                if self.global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        logger.info(
                            f"Validation | "
                            f"Loss: {val_metrics['val_loss']:.4f} | "
                            f"Perplexity: {val_metrics['val_perplexity']:.2f}"
                        )

                        if self.use_wandb:
                            wandb.log(
                                {
                                    "val/loss": val_metrics["val_loss"],
                                    "val/perplexity": val_metrics["val_perplexity"],
                                },
                                step=self.global_step,
                            )

                        # Save best model
                        if val_metrics["val_loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["val_loss"]
                            self.save_checkpoint(is_best=True)
                            logger.info(f"New best validation loss: {self.best_val_loss:.4f}")

                # Save checkpoint
                if self.global_step % self.config.checkpoint_every_n_steps == 0:
                    self.save_checkpoint(is_best=False)

                # Early stopping if max_steps reached
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max_steps={self.config.max_steps}, stopping training")
                    if self.use_wandb:
                        wandb.finish()
                    return

        logger.info("Training completed!")
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.

        Includes averaged model state if SWA/EMA is enabled.

        Args:
            is_best: Whether this is the best model so far
        """
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config.model_dump(),
        }

        # Include averaged model state if SWA/EMA is enabled
        if self.averaged_model is not None:
            checkpoint["averaged_model_state_dict"] = self.averaged_model.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.

        Loads averaged model state if present in checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        # Load averaged model state if present
        if self.averaged_model is not None and "averaged_model_state_dict" in checkpoint:
            self.averaged_model.load_state_dict(checkpoint["averaged_model_state_dict"])
            logger.info("Loaded averaged model state")

        logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
