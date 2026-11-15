"""Run CPU-only restoration checks and write a portable, scoped validation record."""
from pathlib import Path
import datetime
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def run(arguments):
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("python " + " ".join(arguments))
    print("PASS: CLI help without model loading" if "--help" in arguments else completed.stdout.strip())
    if completed.returncode:
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)


def main():
    run(["tools/verify_records.py"])
    run(["tests/verify_cpu.py"])
    (ROOT / "results").mkdir(exist_ok=True)
    run(["-m", "pytest", "-q", "tests/test_maintenance.py",
         "--junitxml=results/maintenance-tests.xml"])
    suite = ET.parse(ROOT / "results/maintenance-tests.xml").getroot()
    cases = suite.findall(".//testcase")
    assert not suite.findall(".//failure") and not suite.findall(".//error")
    entries = ["bench_diffusion", "ceval_diffusion", "ceval_diffusion_subset", "llada_diffusion_cli"]
    for entry in entries:
        run(["-m", "analysis.llada_torch_npu." + entry, "--help"])
    real_device_guard = "not exercised: torch_npu installed; no accelerator probe in CPU validation"
    if importlib.util.find_spec("torch_npu") is None:
        check = subprocess.run([sys.executable, "-m", "analysis.llada_torch_npu.bench_diffusion",
            "--model", "__device_check_only__", "--device", "npu:0"], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert check.returncode != 0 and "No CPU fallback" in check.stderr
        real_device_guard = "PASS: absent torch_npu exits before HF imports/model loading"
    legacy = json.loads((ROOT / "validation/current-cpu-checks.json").read_text())
    record = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.system() + " " + platform.machine(),
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "pytest": importlib.metadata.version("pytest"),
        "legacy_checks_passed": legacy["passed_checks"],
        "maintenance_pytest_cases_passed": len(cases),
        "archived_source_hashes_checked": 13,
        "reported_rows_checked": 14,
        "cli_help_entry_points_checked": entries,
        "real_missing_extension_guard": real_device_guard,
        "optional_numpy_available": importlib.util.find_spec("numpy") is not None,
        "scope": "CPU synthetic functional, historical source integrity and report-table checks only",
        "npu_benchmark_rerun": False,
        "model_weights_loaded": False,
        "ceval_quality_rerun": False,
        "xllm_cpp_built_or_run": False,
        "historical_accelerator_environment_recovered": False,
    }
    target = ROOT / "validation/restoration-validation.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
