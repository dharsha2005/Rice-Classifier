import subprocess
import sys
import unittest
from pathlib import Path


class TestDiagnosticScript(unittest.TestCase):
    def test_diagnostic_script_passes(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "diagnose_model_predictions.py"
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
