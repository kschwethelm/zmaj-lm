"""Unit tests for TrainingConfig"""

from zmaj_lm.config import TrainingConfig


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_initialization(self) -> None:
        config = TrainingConfig()
        assert config.seed == 72
