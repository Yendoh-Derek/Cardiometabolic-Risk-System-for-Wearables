"""
Colab utility functions for environment detection and path handling.

Optional helpers to make Colab notebooks more self-documenting.
All functionality is already available through SSLConfig.from_yaml(),
but these utilities can improve readability.
"""

from pathlib import Path
from typing import Optional


def detect_colab() -> bool:
    """
    Detect if running in Google Colab.
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        from google.colab import drive  # noqa: F401
        return True
    except ImportError:
        return False


def get_drive_path(subdir: str = "") -> Path:
    """
    Get Google Drive path for Colab.
    
    Args:
        subdir: Optional subdirectory within cardiometabolic-risk-colab folder
    
    Returns:
        Path object pointing to Drive location
        
    Raises:
        RuntimeError: If not running in Colab
        
    Example:
        >>> drive_path = get_drive_path("checkpoints/phase5")
        >>> print(drive_path)
        /content/drive/MyDrive/cardiometabolic-risk-colab/checkpoints/phase5
    """
    if not detect_colab():
        raise RuntimeError("Not running in Google Colab")
    
    base_path = Path('/content/drive/MyDrive/cardiometabolic-risk-colab')
    
    if subdir:
        return base_path / subdir
    return base_path


def mount_drive() -> Path:
    """
    Mount Google Drive and return path.
    
    Returns:
        Path object pointing to mounted Drive
        
    Example:
        >>> drive_root = mount_drive()
        >>> assert drive_root.exists()
    """
    from google.colab import drive
    drive.mount('/content/drive')
    return get_drive_path()


def is_gpu_available() -> bool:
    """
    Check if GPU is available in Colab.
    
    Returns:
        True if GPU available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_name() -> Optional[str]:
    """
    Get GPU device name if available.
    
    Returns:
        GPU name (e.g., 'Tesla T4'), or None if no GPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError):
        pass
    return None


def setup_colab_environment() -> dict:
    """
    One-shot setup for Colab environment.
    
    Returns:
        Dictionary with environment info
        
    Example:
        >>> env = setup_colab_environment()
        >>> print(env['is_colab'], env['has_gpu'], env['drive_path'])
    """
    is_colab = detect_colab()
    has_gpu = is_gpu_available()
    drive_path = None
    gpu_name = None
    
    if is_colab:
        mount_drive()
        drive_path = str(get_drive_path())
        gpu_name = get_gpu_name()
    
    return {
        'is_colab': is_colab,
        'has_gpu': has_gpu,
        'gpu_name': gpu_name,
        'drive_path': drive_path,
    }
