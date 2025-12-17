from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Configuration for training a Transformer language model."""

    # Reproducibility
    seed: int = 72

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9  # Adam
    beta2: float = 0.95  # Adam
    grad_clip_norm: float | None = 1.0  # None = no clipping

    # Training loop
    batch_size: int = 32
    num_epochs: int = 10
    max_steps: int | None = None  # If set, overrides num_epochs
    warmup_steps: int = 100  # LR warmup steps
    eval_every_n_steps: int = 500
    checkpoint_every_n_steps: int = 1000
    log_every_n_steps: int = 100

    # LR scheduler
    scheduler_type: str = "cosine"  # e.g., "linear", "cosine"
    min_learning_rate: float = 1e-5

    # Weights & Biases logging
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
