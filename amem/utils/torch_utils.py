"""Utilities for PyTorch device management and GPU support.

This module provides helper functions for device detection, selection,
and configuration - particularly for handling Apple Metal (MPS) with proper
fallbacks for unsupported operations.
"""

import contextlib
import logging
import os
from typing import Any, Literal, Optional, Tuple

import torch
from transformers import AutoModel

# Setup logging
logger = logging.getLogger(__name__)

# Device types
DeviceType = Literal["cpu", "cuda", "mps"]


def setup_mps_fallback() -> bool:
    """Set up MPS fallback for unsupported operations.

    This ensures operations not supported by MPS (like torch.angle)
    will fall back to CPU automatically.

    Returns:
        bool: True if the fallback was set up, False otherwise
    """
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Set PYTORCH_ENABLE_MPS_FALLBACK=1 for MPS fallback")
        return True
    return False


def detect_available_devices() -> Tuple[bool, bool]:
    """Detect which GPU devices are available.

    Returns:
        Tuple[bool, bool]: (has_cuda, has_mps)
    """
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch, "mps") and torch.mps.is_available()

    if has_cuda:
        device_info = f"CUDA available (Device count: {torch.cuda.device_count()}"
        if torch.cuda.device_count() > 0:
            device_info += f", Device: {torch.cuda.get_device_name(0)}"
        device_info += ")"
        logger.info(device_info)

    if has_mps:
        logger.info("Apple Metal (MPS) available")
        # Check if fallback is enabled
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            logger.info("MPS fallback to CPU is enabled for unsupported operations")
        else:
            logger.warning("MPS fallback is NOT enabled! Some operations may fail.")

    if not (has_cuda or has_mps):
        logger.info("No GPU detected, will use CPU")

    return has_cuda, has_mps


def get_device(use_gpu: Optional[bool] = None) -> torch.device:
    """Get the best available device based on preference and availability.

    Args:
        use_gpu: If True, will use GPU if available. If False, will force CPU usage.
                If None, will auto-detect and use GPU if available.

    Returns:
        torch.device: The selected device
    """
    # Ensure MPS fallback is set up
    setup_mps_fallback()

    # Detect available devices
    has_cuda, has_mps = detect_available_devices()

    # Determine if we should use GPU based on user preference and availability
    if use_gpu is None:
        use_gpu_device = has_cuda or has_mps  # Auto-detect any GPU
    else:
        use_gpu_device = use_gpu and (has_cuda or has_mps)

    # Select the appropriate device
    if use_gpu_device:
        if has_cuda:
            device = torch.device("cuda")
            logger.info("Using CUDA GPU")
        elif has_mps:
            device = torch.device("mps")
            logger.info("Using Apple Metal (MPS) GPU")
        else:
            device = torch.device("cpu")
            logger.info("No compatible GPU found, falling back to CPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (GPU usage disabled or unavailable)")

    return device


def safely_move_to_device(model: torch.nn.Module, device: Any) -> torch.nn.Module:
    """Safely move a model to the specified device with error handling.

    Args:
        model: The PyTorch model to move
        device: The target device, which can be a torch.device or a tuple from get_device()

    Returns:
        The model on the target device, or on CPU if moving failed
    """
    try:
        # If device is a tuple (from the old get_device function), use the first element
        if isinstance(device, tuple):
            device = torch.device("cpu")  # Default to CPU if device is a tuple

        return model.to(device)
    except Exception as e:
        logger.error(f"Failed to move model to {device}: {e}")
        logger.info("Falling back to CPU")
        return model.to("cpu")


def fallback_to_cpu_if_needed(func):
    """Decorator to fallback to CPU if an operation fails on the current device.

    This is particularly useful for operations that might not be supported on MPS.

    Args:
        func: The function to wrap with fallback behavior

    Returns:
        A wrapped function that will try CPU if the original device fails
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            device_tensor = None
            # Try to find a tensor argument to determine the device
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type != "cpu":
                    device_tensor = arg
                    break

            if not device_tensor:
                # If we can't find a tensor to determine the device, re-raise
                raise

            logger.warning(f"Operation failed on {device_tensor.device}. Error: {e}")
            logger.info("Attempting fallback to CPU")

            # Convert args to CPU if they're tensors
            cpu_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    cpu_args.append(arg.to("cpu"))
                else:
                    cpu_args.append(arg)

            # Try again on CPU
            result = func(*cpu_args, **kwargs)

            # Move result back to original device if possible
            if isinstance(result, torch.Tensor):
                with contextlib.suppress(Exception):
                    return result.to(device_tensor.device)

            return result

    return wrapper


def load_model(model_name_or_path: str, use_gpu: Optional[bool] = None) -> torch.nn.Module:
    """Load a PyTorch model and move it to the appropriate device.

    Args:
        model_name_or_path: Name or path of the model to load
        use_gpu: If True, will use GPU if available. If False, forces CPU usage.
                If None, auto-detects and uses GPU if available.

    Returns:
        The loaded model on the appropriate device
    """
    try:
        # Get the best device for this system
        device = get_device(use_gpu)

        # Here you would have your loading logic, e.g. with HuggingFace transformers
        # For now, just a placeholder:
        logger.info(f"Loading model: {model_name_or_path}")
        model = AutoModel.from_pretrained(model_name_or_path)

        # Safely move to device
        return safely_move_to_device(model, device)
    except Exception as e:
        logger.error(f"Error loading model {model_name_or_path}: {e}")
        raise
