"""Local isolated subprocess sandbox for running verifier pytest scripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from core.schemas import SandboxResult

TIMEOUT_SECONDS = 15


def run_sandbox(
    verifier_id: str,
    script: str,
    starter_code: str,
    patch_diff: str | None = None,
    *,
    timeout: int = TIMEOUT_SECONDS,
) -> SandboxResult:
    """Run a verifier's pytest script in an isolated tmpdir.

    1. Creates a tempdir
    2. Writes starter code as solution.py
    3. Applies patch_diff (if provided) by overwriting solution.py
    4. Writes the verifier script as test_exploit.py
    5. Runs pytest via subprocess with stripped env and hard timeout
    6. Cleans up
    """
    tmpdir = tempfile.mkdtemp(prefix="veritas_sandbox_")
    try:
        solution_path = Path(tmpdir) / "solution.py"
        solution_path.write_text(starter_code)

        if patch_diff:
            solution_path.write_text(patch_diff)

        test_path = Path(tmpdir) / "test_exploit.py"
        test_path.write_text(script)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_exploit.py", "-v", "--tb=short", "--no-header"],
                cwd=tmpdir,
                env={},
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return SandboxResult(
                verifier_id=verifier_id,
                stdout=result.stdout[-4096:],
                stderr=result.stderr[-4096:],
                exit_code=result.returncode,
                timed_out=False,
                passed=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                verifier_id=verifier_id,
                stdout="",
                stderr=f"Sandbox timed out after {timeout}s",
                exit_code=-1,
                timed_out=True,
                passed=False,
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
