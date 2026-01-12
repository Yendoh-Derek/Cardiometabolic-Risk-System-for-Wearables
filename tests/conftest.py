"""pytest configuration for SSL tests."""
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )


# Fixtures available to all tests
import pytest
import torch

@pytest.fixture(scope="session")
def device():
    """Get available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def signal_batch():
    """Create a dummy signal batch."""
    return torch.randn(4, 1, 75000)


@pytest.fixture
def small_signal_batch():
    """Create a smaller dummy signal batch."""
    return torch.randn(2, 1, 75000)
