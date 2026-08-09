"""
utils/logger.py
===============

This module provides the ExperimentLogger class.

Instead of acting as a simple logger, this class manages the
entire lifecycle of an experiment.

Responsibilities include:

    • Creating experiment directories
    • Creating log files
    • Copying configuration files
    • Recording training metrics
    • Logging checkpoint events

This file will be implemented in multiple stages.
"""

# ==========================================================
# Standard Library Imports
# ==========================================================

# Used for creating console and file loggers.
import logging

# Used for generating timestamps for experiment folders.
from datetime import datetime

# Used for copying configuration files.
import shutil

# pathlib provides an object-oriented way to work with files
# and directories. It is generally preferred over os.path
# because it is easier to read and platform-independent.
from pathlib import Path

# Used only for type hints.
from typing import Iterable, Optional

# PyYAML will later be used when validating configuration
# files and reading experiment settings if needed.
import yaml


class ExperimentLogger:
    """
    Manage one complete machine learning experiment.

    Every experiment receives its own folder containing:

        results/
            experiment_name_timestamp/
                training.log
                metrics.csv
                config/
                checkpoints/

    The trainer interacts only with this class instead of
    manually creating directories or writing log files.
    """

    def __init__(
        self,
        experiment_name: str,
        save_dir: str = "results",
        config_files: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Initialize a new experiment.

        Parameters
        ----------
        experiment_name : str
            Human-readable name for this experiment.

            Example:
                teacher_cifar10
                vanilla_kd_cifar100

        save_dir : str
            Root directory where all experiments are stored.

        config_files : iterable of str, optional
            Paths to YAML configuration files that will later
            be copied into the experiment folder.
        """

        # --------------------------------------------------
        # Store constructor arguments
        # --------------------------------------------------

        self.experiment_name = experiment_name

        # Convert the save directory into a Path object.
        # Path objects are safer and easier to work with
        # than raw strings.
        self.save_dir = Path(save_dir)

        # Convert the config list into a normal list.
        # If None was supplied, use an empty list.
        self.config_files = list(config_files or [])

        # --------------------------------------------------
        # Generate Timestamp
        # --------------------------------------------------
        #
        # Every experiment gets a unique timestamp.
        #
        # Example:
        #
        # 20260807_104530
        #
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --------------------------------------------------
        # Create Experiment Name
        # --------------------------------------------------

        self.run_name = f"{self.experiment_name}_{self.timestamp}"

        # --------------------------------------------------
        # Create Main Experiment Directory
        # --------------------------------------------------

        self.experiment_dir = self.save_dir / self.run_name

        # --------------------------------------------------
        # Create Subdirectories
        # --------------------------------------------------

        self.config_dir = self.experiment_dir / "config"

        self.checkpoint_dir = self.experiment_dir / "checkpoints"

        # --------------------------------------------------
        # Create Directories
        # --------------------------------------------------
        #
        # parents=True
        #
        # Automatically creates missing parent folders.
        #
        # exist_ok=True
        #
        # Prevents an error if the directory already exists.
        #
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # Define Future File Paths
        # --------------------------------------------------

        self.log_file = self.experiment_dir / "training.log"

        self.metrics_file = self.experiment_dir / "metrics.csv"

        # --------------------------------------------------
        # Placeholders
        # --------------------------------------------------
        #
        # These objects will be initialized in Response 1B.
        #
        self.logger = None

        # --------------------------------------------------
        # Setup Logger
        # --------------------------------------------------

        self._setup_file_and_console_logger()

        # --------------------------------------------------
        # Save Configuration Snapshot
        # --------------------------------------------------

        self.save_config_snapshot()

        # --------------------------------------------------
        # Startup Message
        # --------------------------------------------------

        self.logger.info("=" * 70)
        self.logger.info("Experiment initialized.")
        self.logger.info(f"Experiment : {self.run_name}")
        self.logger.info(f"Directory  : {self.experiment_dir}")
        self.logger.info("=" * 70)

    # ------------------------------------------------------
    # Placeholder Methods
    # ------------------------------------------------------
    #
    # These will be fully implemented in Response 1B.
    #
    def _setup_file_and_console_logger(self) -> None:
        """
        Configure the experiment logger.

        This method creates a dedicated logger that writes log messages
        to BOTH:

            1. The terminal (console)
            2. training.log inside the experiment folder

        Every message sent to this logger will automatically appear in
        both locations.

        Using a dedicated logger (instead of Python's root logger)
        prevents conflicts with log messages coming from external
        libraries or future modules.
        """

        # ---------------------------------------------------------
        # Create a Unique Logger
        # ---------------------------------------------------------
        #
        # logging.getLogger(name) returns a logger object.
        #
        # Giving each experiment its own unique name prevents
        # experiments from accidentally sharing handlers.
        #
        self.logger = logging.getLogger(self.run_name)

        # ---------------------------------------------------------
        # Logger Level
        # ---------------------------------------------------------
        #
        # INFO means:
        #
        # INFO
        # WARNING
        # ERROR
        # CRITICAL
        #
        # will all be displayed.
        #
        self.logger.setLevel(logging.INFO)

        # ---------------------------------------------------------
        # Prevent Duplicate Log Messages
        # ---------------------------------------------------------
        #
        # If this logger already has handlers attached (for example,
        # because an experiment was restarted inside the same Python
        # session), remove them before adding new ones.
        #
        # Otherwise every message would appear twice, then three
        # times, and so on.
        #
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # ---------------------------------------------------------
        # Stop Messages from Propagating
        # ---------------------------------------------------------
        #
        # By default, log messages "bubble up" to the parent logger.
        #
        # Setting propagate=False ensures messages are handled ONLY
        # by the handlers we define below.
        #
        self.logger.propagate = False

        # ---------------------------------------------------------
        # Create Log Formatter
        # ---------------------------------------------------------
        #
        # A formatter controls how every log message looks.
        #
        # Example:
        #
        # 2026-08-07 14:52:31 | INFO | Training started
        #
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ---------------------------------------------------------
        # Console Handler
        # ---------------------------------------------------------
        #
        # A StreamHandler writes log messages to the terminal.
        #
        # This allows us to monitor training live.
        #
        console_handler = logging.StreamHandler()

        console_handler.setLevel(logging.INFO)

        console_handler.setFormatter(formatter)

        # ---------------------------------------------------------
        # File Handler
        # ---------------------------------------------------------
        #
        # A FileHandler writes every message into training.log.
        #
        # Even after training finishes, we still have a permanent
        # record of everything that happened.
        #
        file_handler = logging.FileHandler(
            self.log_file,
            mode="a",
            encoding="utf-8",
        )

        file_handler.setLevel(logging.INFO)

        file_handler.setFormatter(formatter)

        # ---------------------------------------------------------
        # Register Handlers
        # ---------------------------------------------------------
        #
        # From now on, every call like
        #
        # self.logger.info(...)
        #
        # automatically writes to BOTH:
        #
        #   terminal
        #   training.log
        #
        self.logger.addHandler(console_handler)

        self.logger.addHandler(file_handler)


    def save_config_snapshot(self) -> None:
        """
        Copy all configuration files used for the experiment into the
        experiment's config directory.

        Missing configuration files are reported as warnings instead
        of terminating the experiment.
        """

        if not self.config_files:
            self.logger.warning("No configuration files were provided.")
            return

        for config_file in self.config_files:

            source_path = Path(config_file)

            if not source_path.exists():
                self.logger.warning(
                    f"Configuration file not found: {source_path}"
                )
                continue

            destination_path = self.config_dir / source_path.name

            try:
                shutil.copy2(source_path, destination_path)

                self.logger.info(
                    f"Configuration snapshot saved: {destination_path.name}"
                )

            except Exception as exc:
                self.logger.error(
                    f"Failed to copy configuration file "
                    f"'{source_path}': {exc}"
                )

    def info(self, message: str) -> None:
        """
        Log an informational message.
        """

        self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        Log a warning message.
        """

        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Log an error message.
        """

        self.logger.error(message)

    def log_epoch(
    self,
    epoch: int,
    total_epochs: int,
    metrics: dict,
    lr: float,
    elapsed_time: float,
) -> None:
        """
        Log one epoch of training metrics in a standardized format.
        """

        message = (
            f"Epoch [{epoch:03d}/{total_epochs}] | "
            f"Train Loss: {metrics.get('train_loss', 0.0):.4f} | "
            f"Val Loss: {metrics.get('val_loss', 0.0):.4f} | "
            f"Top-1: {metrics.get('top1', 0.0):.2f}% | "
            f"Top-5: {metrics.get('top5', 0.0):.2f}% | "
            f"LR: {lr:.6f} | "
            f"Time: {elapsed_time:.2f}s"
        )

        self.logger.info(message)
