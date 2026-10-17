#!/usr/bin/env python3
"""Run checksum-verified cases with bounded resources and fresh evidence roots."""
import argparse
import hashlib
import importlib.util
from importlib import metadata
import re
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import tempfile
import shutil
import stat
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "PyZMAT/src")]

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def cases():
    registry = json.loads((ROOT / "tests/meta.json").read_text())
    result = []
    for row in registry["cases"]:
        folder = ROOT / row["path"]
        case = json.loads((folder / "case.json").read_text())
        required = {"schema_version", "test_id", "package", "level", "runner", "limits", "toolchain", "tolerance_class", "status", "reference"}
        if not required.issubset(case) or case["schema_version"] != 2:
            raise ValueError("Invalid case contract")
        if case["status"] not in {"prepared", "baseline_accepted", "historical_reference", "reference_needs_rerun", "retired"}:
            raise ValueError("Invalid lifecycle status")
        if case["runner"]["adapter"] != "pytest_case":
            raise ValueError("Unsupported adapter")
        if case["tolerance_class"] not in {"T0", "T1", "T2", "T3"}:
            raise ValueError("Invalid tolerance class")
        if any(case["limits"][name] <= 0 for name in ("wall_s", "cpu_s", "memory_mb", "threads")):
            raise ValueError("Resource limits must be positive")
        if case["level"] == "L1" and (case["limits"]["wall_s"] > 60 or case["limits"]["cpu_s"] > 45 or case["limits"]["threads"] != 1):
            raise ValueError("L1 limits exceed framework policy")
        if not (folder / case["reference"]["path"]).is_file():
            raise ValueError("Missing reference contract")
        for field in ("test_id", "level", "status", "package"):
            if row[field] != case[field]:
                raise ValueError("Registry/contract mismatch: " + row["test_id"])
        if case["level"] not in ("L0", "L1", "L2", "L3") or not case["runner"]["args"]["nodes"]:
            raise ValueError("Invalid case: " + case["test_id"])
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != expected:
                raise ValueError("Fixture checksum mismatch: " + name)
        result.append(case)
    if len({c["test_id"] for c in result}) != len(result):
        raise ValueError("Duplicate case ID")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("L0", "L1", "L2", "L3"), default="L0")
    parser.add_argument("--test-id", action="append")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-all", action="store_true", help="Skips make this invocation incomplete (exit 2)")
    args = parser.parse_args()
    all_cases = cases()
    if args.validate:
        print(f"Validated {len(all_cases)} case contracts and checksums")
        return 0
    selected = [c for c in all_cases if c["test_id"] in args.test_id] if args.test_id else [c for c in all_cases if c["level"] == args.level]
    if args.test_id and set(args.test_id) != {c["test_id"] for c in selected}:
        parser.error("Unknown test ID")
    if not selected:
        parser.error("No cases selected")
    if args.list:
        for case in selected:
            print(case["test_id"], case["level"], case["status"])
        return 0
    if not args.out:
        parser.error("--out must name a fresh evidence directory")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for folder in ("src", "tests", "tools", "scripts") for p in (ROOT / folder).rglob("*.py")}
    env = os.environ.copy()
    env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT.parent / "PyZMAT/src"), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
    external_hashes = {}
    for key in ("LAM_MODEL", "ML_LAMS_GPU_REGRESSION_MODEL"):
        if env.get(key) and Path(env[key]).is_file():
            external_hashes[key] = digest(env[key])
    results = []
    for case in selected:
        work = out / case["test_id"]
        work.mkdir()
        provenance = dict(source_revision=revision, source_sha256=source_hashes, python=sys.executable, host=os.uname().nodename, date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), thread_count=1)
        provenance.update(packages=packages, external_sha256=external_hashes)
        (work / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        row = dict(test_id=case["test_id"], provenance=provenance, scientific={}, performance={})
        missing = [m for m in case["toolchain"].get("python_modules", []) if importlib.util.find_spec(m) is None]
        missing += [name for name in case["toolchain"].get("environment", []) if not env.get(name)]
        if case["level"] == "L3" and not env.get("PBS_JOBID"):
            missing.append("PBS_JOBID (allocated campaign required)")
        if missing:
            row["scientific"] = dict(status="SKIP", reason="Missing " + ", ".join(missing))
        else:
            child_env = dict(env, TIER_LEVEL=case["level"])
            if not case["toolchain"].get("gpu_required"):
                child_env["CUDA_VISIBLE_DEVICES"] = ""
            local_tmp = Path(tempfile.mkdtemp(prefix="tier-", dir="/tmp"))
            xml = work / "pytest.xml"
            command = ["/usr/bin/time", "-v", "-o", str(work / "time.txt"), sys.executable, "-m", "pytest", "-q", "-ra", "--junitxml=" + str(xml), "--basetemp=" + str(local_tmp), *case["runner"]["args"]["nodes"]]
            if not Path("/usr/bin/time").is_file():
                command = [sys.executable, str(ROOT / "tools/time_case.py"), str(work / "time.json"), *command[4:]]
            def limits():
                cpu = int(case["limits"]["cpu_s"])
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            started = time.monotonic()
            with (work / "output.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=limits)
                timed_out = False
                try:
                    code = process.wait(timeout=case["limits"]["wall_s"])
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
            def omit_special(folder, names):
                return [name for name in names if not (Path(folder, name).is_file() or Path(folder, name).is_dir() or Path(folder, name).is_symlink())]
            shutil.copytree(local_tmp, work / "artifacts", symlinks=True, ignore=omit_special)
            shutil.rmtree(local_tmp)
            row["performance"]["wall_s"] = time.monotonic() - started
            counts = dict(tests=0, failures=0, errors=0, skipped=0)
            if xml.exists():
                suites = ET.parse(xml).getroot().iter("testsuite")
                for suite in suites:
                    for key in counts:
                        counts[key] += int(suite.get(key, 0))
            status = "FAIL" if timed_out or code or not counts["tests"] else ("SKIP" if counts["skipped"] else "PASS")
            row["scientific"] = dict(status=status, exit_code=code, timed_out=timed_out, **counts)
            timing = work / "time.txt"
            if timing.exists():
                row["performance"]["time_v"] = timing.read_text()
                rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", timing.read_text())
                if rss:
                    row["performance"]["peak_rss_kb"] = int(rss.group(1))
                    if int(rss.group(1)) > case["limits"]["memory_mb"] * 1024:
                        row["scientific"].update(status="FAIL", reason="Peak RSS exceeds contract")
            if (work / "time.json").exists():
                row["performance"].update(json.loads((work / "time.json").read_text()))
                if row["performance"]["peak_rss_kb"] > case["limits"]["memory_mb"] * 1024:
                    row["scientific"].update(status="FAIL", reason="Peak RSS exceeds contract")
            provenance["command"] = command
        results.append(row)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(case["test_id"], row["scientific"]["status"], flush=True)
    if any(r["scientific"]["status"] == "FAIL" for r in results):
        return 1
    return 2 if args.require_all and any(r["scientific"]["status"] == "SKIP" for r in results) else 0

if __name__ == "__main__":
    raise SystemExit(main())
