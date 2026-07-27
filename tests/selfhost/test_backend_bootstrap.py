#!/usr/bin/env python3
"""Regression test for source-to-source corec/corearch bootstrapping."""

import os
import shutil
import stat
import subprocess
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
BUILD = BASE / "build"
COREC = Path(os.environ.get("CORE_BOOTSTRAP_COREC", BUILD / "corec")).resolve()
COREARCH = Path(os.environ.get("CORE_BOOTSTRAP_COREARCH", COREC.with_name("corearch"))).resolve()
COMPILER_SOURCE = BASE / "src" / "compiler"
BACKEND_SOURCE = BASE / "src" / "arch" / "linux" / "ld"
TEST_BUILD = BUILD / "test_backend_bootstrap"


def run_checked(args: list[Path | str], label: str, env: dict[str, str]) -> bool:
    command = ["nice", "-n", "19", *map(str, args)]
    result = subprocess.run(
        command,
        cwd=BASE,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode == 0:
        print(f"[PASS] {label}")
        return True
    print(f"[FAIL] {label}: exit {result.returncode}")
    print(result.stdout)
    print(result.stderr)
    return False


def make_executable(path: Path) -> bool:
    if not path.exists():
        print(f"[FAIL] expected output was not created: {path}")
        return False
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return True


def build_stage(
    previous_corec: Path,
    stage_dir: Path,
    stage_number: int,
    env: dict[str, str],
) -> bool:
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_corec = stage_dir / "corec"
    stage_corearch = stage_dir / "corearch"

    if not run_checked(
        [
            previous_corec,
            "build",
            COMPILER_SOURCE,
            "-o",
            stage_corec,
            "--static",
            "-O",
            "0",
        ],
        f"stage {stage_number - 1} builds corec stage {stage_number}",
        env,
    ) or not make_executable(stage_corec):
        return False

    if not run_checked(
        [
            previous_corec,
            "build",
            BACKEND_SOURCE,
            "-o",
            stage_corearch,
            "--static",
            "-O",
            "0",
        ],
        f"stage {stage_number - 1} builds corearch stage {stage_number}",
        env,
    ) or not make_executable(stage_corearch):
        return False

    return True


def run_smoke(stage_dir: Path, stage_number: int, source: Path, env: dict[str, str]) -> bool:
    output = stage_dir / "smoke"
    if not run_checked(
        [
            stage_dir / "corec",
            "build",
            source,
            "-o",
            output,
            "--static",
            "-O",
            "0",
        ],
        f"stage {stage_number} compiler pair builds smoke ELF",
        env,
    ) or not make_executable(output):
        return False

    result = subprocess.run([str(output)], cwd=BASE, timeout=30)
    if result.returncode != 7:
        print(f"[FAIL] stage {stage_number} smoke: expected 7, got {result.returncode}")
        return False
    print(f"[PASS] stage {stage_number} smoke ELF returns 7")
    return True


def main() -> int:
    if not COREC.exists() or not COREARCH.exists():
        print(f"bootstrap compiler pair is missing: {COREC}, {COREARCH}")
        return 1
    if COREC.parent != COREARCH.parent:
        print("bootstrap corec and corearch must be in the same directory")
        return 1

    env = os.environ.copy()
    env["PATH"] = str(BUILD) + os.pathsep + env.get("PATH", "")
    shutil.rmtree(TEST_BUILD, ignore_errors=True)
    TEST_BUILD.mkdir(parents=True)

    try:
        stages = [TEST_BUILD / f"stage{number}" for number in range(1, 4)]
        previous_corec = COREC
        for number, stage_dir in enumerate(stages, start=1):
            if not build_stage(previous_corec, stage_dir, number, env):
                return 1
            previous_corec = stage_dir / "corec"

        stage2_corearch = stages[1] / "corearch"
        stage3_corearch = stages[2] / "corearch"
        if stage2_corearch.read_bytes() != stage3_corearch.read_bytes():
            print("[FAIL] source-regenerated corearch stages 2 and 3 are not byte-identical")
            return 1
        print("[PASS] source-regenerated corearch stages 2 and 3 are byte-identical")

        smoke_source = TEST_BUILD / "smoke.cr"
        smoke_source.write_text("fn main() -> int { return 7; }\n", encoding="utf-8")
        for number, stage_dir in enumerate(stages, start=1):
            if not run_smoke(stage_dir, number, smoke_source, env):
                return 1

        smoke_o2 = stages[2] / "smoke_o2"
        if not run_checked(
            [
                stages[2] / "corec",
                "build",
                smoke_source,
                "-o",
                smoke_o2,
                "--static",
                "-O",
                "2",
            ],
            "stage 3 compiler pair builds O2 smoke ELF",
            env,
        ) or not make_executable(smoke_o2):
            return 1
        result = subprocess.run([str(smoke_o2)], cwd=BASE, timeout=30)
        if result.returncode != 7:
            print(f"[FAIL] stage 3 O2 smoke: expected 7, got {result.returncode}")
            return 1
        print("[PASS] stage 3 O2 smoke ELF returns 7")
        return 0
    finally:
        shutil.rmtree(TEST_BUILD, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
