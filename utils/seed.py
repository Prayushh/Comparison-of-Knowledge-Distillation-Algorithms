"""
utils/seed.py
=============

This module provides utility functions for making machine learning
experiments as reproducible as possible.

Why is this necessary?

Training a neural network involves many random operations, including:

    • Random weight initialization
    • Random data shuffling
    • Random data augmentation
    • Random dropout masks
    • CUDA kernel selection
    • Python's internal hashing

If these sources of randomness are not controlled, two runs of the
same experiment can produce different results.

By setting all random seeds consistently, we greatly improve
experiment reproducibility.

NOTE:
Some CUDA operations are inherently nondeterministic due to hardware
implementation. This utility minimizes randomness as much as PyTorch
currently allows.
"""

# ==========================================================
# Standard Library Imports
# ==========================================================

# Used for Python's built-in random number generator.
import random

# Used to modify environment variables.
import os

# ==========================================================
# Third-Party Imports
# ==========================================================

# NumPy has its own independent random generator.
import numpy as np

# PyTorch maintains separate CPU and GPU random generators.
import torch


def set_seed(seed: int = 42, verbose: bool = False) -> None:
    """
    Set random seeds across the entire project.

    Parameters
    ----------
    seed : int, optional
        Seed value used for all random number generators.

    verbose : bool, optional
        If True, print a confirmation message.
        By default this is False because logging should be handled
        by the project's ExperimentLogger.

    Returns
    -------
    None

    Notes
    -----
    This function should be called once near the beginning of every
    training script before creating:

        • datasets
        • dataloaders
        • models
        • optimizers
    """

    # ------------------------------------------------------
    # Python Random Module
    # ------------------------------------------------------
    #
    # Python's built-in "random" module is used by many
    # libraries internally.
    #
    # Example:
    #
    # random.randint(...)
    # random.shuffle(...)
    #
    random.seed(seed)

    # ------------------------------------------------------
    # Python Hash Seed
    # ------------------------------------------------------
    #
    # Python dictionaries and sets internally rely on hash
    # functions.
    #
    # Fixing PYTHONHASHSEED improves reproducibility of
    # operations involving hashing.
    #
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ------------------------------------------------------
    # NumPy Random Generator
    # ------------------------------------------------------
    #
    # NumPy maintains an independent random state.
    #
    # This controls operations such as:
    #
    # np.random.rand(...)
    # np.random.randn(...)
    # np.random.shuffle(...)
    #
    np.random.seed(seed)

    # ------------------------------------------------------
    # PyTorch CPU Random Generator
    # ------------------------------------------------------
    #
    # Controls random operations performed on CPU tensors.
    #
    torch.manual_seed(seed)

    # ------------------------------------------------------
    # CUDA Random Generator
    # ------------------------------------------------------
    #
    # If CUDA is available, seed every GPU.
    #
    # manual_seed_all() is useful even if only one GPU is
    # currently installed because it also supports future
    # multi-GPU training without code changes.
    #
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------
    # CuDNN Deterministic Mode
    # ------------------------------------------------------
    #
    # CuDNN contains several optimized implementations of
    # convolution operations.
    #
    # Some of these implementations are extremely fast but
    # not perfectly deterministic.
    #
    # Setting deterministic=True forces PyTorch to use only
    # deterministic algorithms whenever possible.
    #
    torch.backends.cudnn.deterministic = True

    # ------------------------------------------------------
    # Disable CuDNN Benchmark Mode
    # ------------------------------------------------------
    #
    # benchmark=True allows CuDNN to benchmark multiple
    # convolution algorithms and automatically choose the
    # fastest one.
    #
    # Although this improves performance, different algorithms
    # may be selected across runs.
    #
    # Since our goal is reproducibility instead of maximum
    # speed, we disable benchmarking.
    #
    torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------
    # Optional Strict Deterministic Mode
    # ------------------------------------------------------
    #
    # Uncommenting the line below forces PyTorch to raise an
    # error whenever a nondeterministic operation is used.
    #
    # This is useful for debugging but can sometimes prevent
    # certain models from training.
    #
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    if verbose:
        print(f"[Seed] Random seed set to {seed}")