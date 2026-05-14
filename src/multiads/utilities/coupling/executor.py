"""Subprocess execution utilities for CFD-CSM coupling.

This module provides utilities for running external solvers (DUST, etc.)
as subprocesses with proper logging and error handling.
"""

from __future__ import annotations

import logging
import os
import subprocess as sp
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def _log_cmd_output(
    file_name: Path | str,
    cmd: sp.CompletedProcess,
    name: str,
    mode: str = "a",
) -> None:
    """Log command output to a file.

    Args:
        file_name: Path to the log file.
        cmd: Completed subprocess result.
        name: Name of the command for error messages.
        mode: File open mode ('w' or 'a').
    """
    with Path(file_name).open(mode) as f:
        for line in cmd.stdout.splitlines():
            f.write(f"INFO: {line}\n")
        for line in cmd.stderr.splitlines():
            f.write(f"ERROR: {line}\n")
        if cmd.returncode != 0:
            raise RuntimeError(f"could not run '{name}'")


def run_command(
    command: str | list[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    n_threads: int | None = None,
    log_file: Path | str | None = None,
    name: str = "command",
    timeout: float | None = None,
) -> sp.CompletedProcess[str]:
    """Run an external command as a subprocess.

    Args:
        command: Command to execute (string or list).
        env: Environment variables.
        cwd: Working directory.
        n_threads: Number of threads (sets OMP_NUM_THREADS).
        log_file: Path to log file. If None, uses '{name}.log'.
        name: Name for logging.
        timeout: Command timeout in seconds.

    Returns:
        CompletedProcess result.

    Raises:
        RuntimeError: If command fails.
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    if n_threads is not None:
        run_env["OMP_NUM_THREADS"] = str(n_threads)

    main_dir = Path.cwd()
    if cwd is not None:
        os.chdir(Path(cwd))

    try:
        if isinstance(command, str):
            cmd_list = command
            shell = True
        else:
            cmd_list = command
            shell = False

        logger.info(f"Running {name}...")
        result = sp.run(
            cmd_list,
            capture_output=True,
            text=True,
            env=run_env,
            shell=shell,
            timeout=timeout,
        )

        if log_file is None:
            log_file = Path(f"{name}.log")
        _log_cmd_output(log_file, result, name, mode="a")

        os.chdir(main_dir)
        return result

    except sp.TimeoutExpired as e:
        os.chdir(main_dir)
        raise RuntimeError(f"{name} timed out after {timeout}s") from e
    except Exception as e:
        os.chdir(main_dir)
        raise RuntimeError(f"Failed to run {name}: {e}") from e


def run_dust(
    command: str = "dust",
    n_threads: int = 10,
    work_dir: Path | str = ".",
    log_file: Path | str | None = None,
) -> sp.CompletedProcess[str]:
    """Run DUST CFD solver.

    Args:
        command: DUST command to run.
        n_threads: Number of OpenMP threads.
        work_dir: Working directory.
        log_file: Optional log file path.

    Returns:
        CompletedProcess result.
    """
    return run_command(
        command,
        n_threads=n_threads,
        cwd=work_dir,
        log_file=log_file,
        name="dust",
    )


def run_dust_pre(
    n_threads: int = 10,
    work_dir: Path | str = ".",
    log_file: Path | str | None = None,
) -> sp.CompletedProcess[str]:
    """Run DUST preprocessor.

    Args:
        n_threads: Number of OpenMP threads.
        work_dir: Working directory.
        log_file: Optional log file path.

    Returns:
        CompletedProcess result.
    """
    return run_dust("dust_pre", n_threads, work_dir, log_file)


def run_lagrange(
    command: str = "lagrange",
    n_threads: int = 1,
    work_dir: Path | str = ".",
    log_file: Path | str | None = None,
) -> sp.CompletedProcess[str]:
    """Run Lagrange structural solver.

    Args:
        command: Lagrange command to run.
        n_threads: Number of threads.
        work_dir: Working directory.
        log_file: Optional log file path.

    Returns:
        CompletedProcess result.
    """
    return run_command(
        command,
        n_threads=n_threads,
        cwd=work_dir,
        log_file=log_file,
        name="lagrange",
    )


def setup_logging(
    log_file: str | Path = "cfd_csm.log",
    level: int = logging.INFO,
) -> None:
    """Configure logging for coupling simulations.

    Args:
        log_file: Path to log file.
        level: Logging level.
    """
    logging.basicConfig(
        filename=str(log_file),
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=level,
    )
