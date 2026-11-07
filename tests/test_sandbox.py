"""Tests for infra/sandbox.py — isolated subprocess execution."""

from infra.sandbox import run_sandbox


TRIVIAL_CODE = "def add(a, b):\n    return a + b\n"


class TestSandboxPassing:
    def test_passing_script(self):
        script = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        result = run_sandbox("V_test", script, TRIVIAL_CODE)
        assert result.passed is True
        assert result.exit_code == 0
        assert result.timed_out is False
        assert "passed" in result.stdout.lower()


class TestSandboxFailing:
    def test_failing_script(self):
        script = "from solution import add\n\ndef test_add_wrong():\n    assert add(2, 3) == 99\n"
        result = run_sandbox("V_test", script, TRIVIAL_CODE)
        assert result.passed is False
        assert result.exit_code != 0
        assert result.timed_out is False

    def test_timeout(self):
        script = "import time\n\ndef test_slow():\n    time.sleep(30)\n"
        result = run_sandbox("V_test", script, TRIVIAL_CODE, timeout=2)
        assert result.timed_out is True
        assert result.passed is False


class TestSandboxPatchApplied:
    def test_patch_overrides_starter(self):
        bad_code = "def add(a, b):\n    return 0\n"
        good_code = "def add(a, b):\n    return a + b\n"
        script = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        result = run_sandbox("V_ast", script, bad_code, patch_diff=good_code)
        assert result.passed is True
