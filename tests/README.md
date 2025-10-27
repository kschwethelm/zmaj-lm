# Tests

This directory contains all tests for the zmaj-lm project.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures and configuration
├── test_environment.py      # Environment and setup verification tests
├── unit/                    # Unit tests for individual components
│   └── __init__.py
└── integration/             # Integration tests for full workflows
    └── __init__.py
```

## Running Tests

### Run all tests
```bash
uv run pytest
```

### Run specific test file
```bash
uv run pytest tests/test_environment.py
```

### Run tests with verbose output
```bash
uv run pytest -v
```

### Run only fast tests (exclude slow tests)
```bash
uv run pytest -m "not slow"
```

### Run tests in parallel
```bash
uv run pytest -n auto
```

### Run with coverage report
```bash
uv run pytest --cov=zmaj_lm --cov-report=html
```

## Test Categories

Tests are organized by markers:

- **slow**: Long-running tests (e.g., full training runs)
- **integration**: Tests that check multiple components working together
- **GPU tests**: Automatically skipped when no GPU is available

## GPU Testing

GPU-specific tests will automatically skip when no CUDA device is detected. To verify GPU setup:

```bash
uv run pytest tests/test_environment.py::test_jax_cuda_available -v
```

## Writing Tests

### Unit Tests

Place unit tests for specific modules in `tests/unit/`:

```python
# tests/unit/test_model.py
def test_transformer_forward():
    # Test individual components
    pass
```

### Integration Tests

Place integration tests in `tests/integration/`:

```python
# tests/integration/test_training.py
@pytest.mark.slow
def test_full_training_pipeline():
    # Test complete workflows
    pass
```

### Using Fixtures

Common fixtures are available from `conftest.py`:

```python
def test_with_fixtures(rng_key, has_gpu):
    if has_gpu:
        # GPU-specific testing
        pass
```
