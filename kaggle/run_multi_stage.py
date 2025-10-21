"""
Orchestrate multiple RL stacks sequentially in a Kaggle notebook.

Each stage executes inside a throwaway virtual environment so that its
dependencies never leak into the global site-packages. The orchestrator records
logs and artifacts for every run and optionally zips the directory for easy
download from notebook environments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import venv


def _get_site_packages_path(python_executable: Path) -> Path:
    """Return the site-packages directory for the provided Python interpreter."""
    result = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_stage_env(),
    )
    return Path(result.stdout.strip())


def _write_pip_entrypoints(python_executable: Path) -> None:
    """Ensure pip console scripts exist inside the virtual environment."""

    version_result = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_stage_env(),
    )
    version = version_result.stdout.strip()
    major, _, minor = version.partition(".")

    script_body = (
        f"#!{python_executable}\n"
        "from runpy import run_module\n"
        "if __name__ == '__main__':\n"
        "    run_module('pip', run_name='__main__')\n"
    )

    bin_dir = python_executable.parent
    candidate_names = ["pip", "pip3", f"pip{major}", f"pip{major}.{minor}" if minor else None]
    script_names: list[str] = []
    for candidate in candidate_names:
        if candidate and candidate not in script_names:
            script_names.append(candidate)

    for name in script_names:
        script_path = bin_dir / name
        script_path.write_text(script_body, encoding="utf-8")
        script_stat = script_path.stat()
        script_path.chmod(script_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _fallback_copy_pip(python_executable: Path, env_dir: Path, error: subprocess.CalledProcessError) -> None:
    """Copy pip and its metadata from the orchestrator environment into the venv."""

    print(
        "[orchestrator] ensurepip failed with exit code"
        f" {error.returncode}; attempting manual pip bootstrap."
    )

    site_packages = _get_site_packages_path(python_executable)

    try:
        import pip  # type: ignore
    except Exception as import_error:  # pragma: no cover - defensive logging
        raise RuntimeError(
            "Failed to import pip from the orchestrator environment while "
            "attempting a manual bootstrap"
        ) from import_error

    pip_module_path = Path(pip.__file__).resolve()
    if pip_module_path.name == "__init__.py":
        pip_source = pip_module_path.parent
    else:
        pip_source = pip_module_path

    source_root = pip_source.parent

    if pip_source.is_dir():
        shutil.copytree(pip_source, site_packages / pip_source.name, dirs_exist_ok=True)
    else:
        shutil.copy2(pip_source, site_packages / pip_source.name)

    dist_info_candidates = sorted(source_root.glob("pip-*.dist-info"))
    if not dist_info_candidates:
        raise RuntimeError("Unable to locate pip dist-info directory for manual bootstrap")

    dist_info_source = dist_info_candidates[-1]
    shutil.copytree(dist_info_source, site_packages / dist_info_source.name, dirs_exist_ok=True)

    _write_pip_entrypoints(python_executable)

    verification = subprocess.run(
        [str(python_executable), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        env=_stage_env(),
    )
    if verification.returncode != 0:
        raise RuntimeError(
            "Manual pip bootstrap failed; pip is still unavailable in the virtualenv"
        )



ORCHESTRATOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORCHESTRATOR_DIR.parent


def _augment_pythonpath(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return environment vars with repo root prepended to PYTHONPATH."""

    base_env = os.environ.copy() if env is None else env.copy()
    repo_path = str(REPO_ROOT)
    existing = base_env.get("PYTHONPATH")
    if existing:
        entries = existing.split(os.pathsep)
        if repo_path not in entries:
            base_env["PYTHONPATH"] = os.pathsep.join([repo_path, *entries])
    else:
        base_env["PYTHONPATH"] = repo_path
    return base_env


def _stage_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Augment environment variables for stage subprocesses."""

    stage_env = _augment_pythonpath(env)

    wheelhouse = Path("/kaggle/working/wheelhouse")
    try:
        if wheelhouse.parent.exists():
            wheelhouse.mkdir(parents=True, exist_ok=True)
            stage_env.setdefault("PIP_FIND_LINKS", str(wheelhouse))
            stage_env.setdefault("PIP_NO_INDEX", "1")
    except Exception:
        pass  # Non-Kaggle environments may lack permission; best-effort.

    cache_dir = Path("/kaggle/working/.cache/pip")
    try:
        if cache_dir.parent.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            stage_env.setdefault("PIP_CACHE_DIR", str(cache_dir))
    except Exception:
        pass

    return stage_env


def _detect_python_binary(venv_dir: Path) -> Path:
    if os.name == "nt":
        candidate = venv_dir / "Scripts" / "python.exe"
    else:
        candidate = venv_dir / "bin" / "python"
    if not candidate.exists():
        raise RuntimeError(f"Missing python binary in virtualenv: {candidate}")
    return candidate


@dataclass(frozen=True)
class StageDefinition:
    name: str
    requirements: Sequence[str]
    entrypoint: Path
    uninstall: Sequence[str]
    description: str
    bootstrap: Sequence[str] = ()
    # ``uninstall`` is retained for compatibility with existing configs but no longer
    # used now that each stage executes inside an isolated virtual environment.


def discover_artifact_root(custom_root: Path | None) -> Path:
    if custom_root is not None:
        return custom_root
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.exists():
        return kaggle_root / "multi_stage_artifacts"
    return Path.cwd() / "multi_stage_artifacts"


def package_artifacts(artifact_root: Path) -> Path:
    """Create a downloadable zip of the entire artifact tree."""
    zip_base = artifact_root.parent / artifact_root.name
    zip_path = zip_base.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    created = shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=artifact_root.parent,
        base_dir=artifact_root.name,
    )
    return Path(created)


def create_virtualenv(stage_name: str, venv_root: Path) -> tuple[Path, Path]:
    env_dir = venv_root / stage_name
    if env_dir.exists():
        shutil.rmtree(env_dir)
    builder = venv.EnvBuilder(with_pip=False, clear=True)
    builder.create(env_dir)
    python_executable = _detect_python_binary(env_dir)
    python_env = _stage_env()
    try:
        subprocess.run(
            [
                str(python_executable),
                "-m",
                "ensurepip",
                "--upgrade",
                "--default-pip",
            ],
            check=True,
            env=python_env,
        )
    except subprocess.CalledProcessError as error:
        _fallback_copy_pip(python_executable, env_dir, error)
    subprocess.run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        check=True,
        env=python_env,
    )
    return python_executable, env_dir


def pip_install(
    python_executable: Path,
    requirements: Sequence[str],
    *,
    bootstrap: Sequence[str] = (),
) -> None:
    python_env = _stage_env()
    if bootstrap:
        bootstrap_cmd = [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--upgrade",
            *bootstrap,
        ]
        subprocess.run(bootstrap_cmd, check=True, env=python_env)
    if not requirements:
        return
    cmd = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--no-input",
        "--upgrade",
        *requirements,
    ]
    subprocess.run(cmd, check=True, env=python_env)


def pip_check(python_executable: Path) -> None:
    result = subprocess.run(
        [str(python_executable), "-m", "pip", "check"],
        capture_output=True,
        text=True,
        env=_stage_env(),
    )
    if result.returncode != 0:
        print("[orchestrator] pip check reported issues (continuing):")
        print(result.stdout.strip())
        print(result.stderr.strip())


def run_stage(
    stage: StageDefinition,
    output_dir: Path,
    python_executable: Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[str, str, float]:
    """Execute stage entrypoint synchronously and capture logs."""
    stage_dir = output_dir / stage.name
    if stage_dir.exists():
        if stage_dir.is_dir():
            shutil.rmtree(stage_dir)
        else:
            stage_dir.unlink()
    stage_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    requirements_path = stage_dir / "requirements.txt"
    requirements_path.write_text("\n".join(stage.requirements) + "\n", encoding="utf-8")

    command = [
        str(python_executable),
        str(stage.entrypoint),
        "--output-dir",
        str(stage_dir),
    ]

    env = _stage_env()
    pythonpath_entries = [str(REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if extra_env:
        env.update(extra_env)

    start = time.time()
    result = subprocess.run(
        command,
        cwd=stage.entrypoint.parent,
        text=True,
        capture_output=True,
        env=env,
    )
    duration = time.time() - start

    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage {stage.name} failed (exit={result.returncode}); see {stderr_path}"
        )

    return stdout_path.as_posix(), stderr_path.as_posix(), duration


def build_argument_parser(stage_registry: dict[str, StageDefinition]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple RL stacks sequentially with dependency isolation. "
            "By default all registered stages execute in the predefined order."
        )
    )
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        help="Stage name to execute (can be provided multiple times).",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        help="Directory for collected artifacts (defaults to Kaggle working directory).",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Keep packages installed after each stage (not recommended).",
    )
    parser.add_argument(
        "--skip-zip",
        action="store_true",
        help="Skip creating a zip archive for the collected artifacts.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print registered stages and exit.",
    )
    return parser


def list_stages(stage_registry: dict[str, StageDefinition]) -> None:
    payload = [
        {"name": stage.name, "description": stage.description}
        for stage in stage_registry.values()
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    stage_registry: dict[str, StageDefinition] = {
        "sb3_kaggle": StageDefinition(
            name="sb3_kaggle",
            requirements=[
                "gymnasium==0.29.0",
                "stable-baselines3==2.1.0",
                "sb3-contrib==2.1.0",
                "kaggle-environments==1.18.0",
            ],
            entrypoint=Path(__file__).parent / "stages" / "sb3_kaggle.py",
            uninstall=[
                "gymnasium",
                "stable-baselines3",
                "sb3-contrib",
                "kaggle-environments",
            ],
            description="Stable-Baselines3 PPO smoke test compatible with Kaggle stack.",
        ),
        "dopamine": StageDefinition(
            name="dopamine",
            requirements=[
                "gymnasium>=1.0.0",
                "dopamine-rl==4.1.2",
            ],
            entrypoint=Path(__file__).parent / "stages" / "dopamine.py",
            uninstall=[
                "gymnasium",
                "dopamine-rl",
            ],
            description="Random-policy dopamine rollouts on Gymnasium 1.x.",
        ),
        "sb3_leadlag": StageDefinition(
            name="sb3_leadlag",
            requirements=[
                # Minimal stack for production RL on LeadLag (SB3 2.x + Gymnasium)
                "numpy>=1.23,<2.0",
                "pandas>=1.5,<2.2.3",
                "scipy>=1.10,<1.16",
                "scikit-learn>=1.1,<1.7",
                "matplotlib>=3.6,<3.11",
                "tqdm>=4.64",
                "pyyaml>=6.0",
                "gymnasium==0.29.1",
                "stable-baselines3==2.1.0",
                "sb3-contrib==2.1.0",
                "torch>=2.1,<2.7",
            ],
            entrypoint=Path(__file__).parent / "stages" / "sb3_leadlag.py",
            uninstall=[
                "gym",
                "gymnasium",
                "stable-baselines3",
                "sb3-contrib",
                "torch",
                "dopamine-rl",
            ],
            description="Train PPO/variants on LeadLag env via training.run_rl with overrides.",
        ),
        "full_suite": StageDefinition(
            name="full_suite",
            requirements=[
                # Use the project's lean stack; RL deps are handled in sb3_leadlag stage
                "-r",
                str(REPO_ROOT / "requirements-kaggle.txt"),
            ],
            entrypoint=Path(__file__).parent / "stages" / "full_suite.py",
            uninstall=[
                "gym",
                "gymnasium",
                "stable-baselines3",
                "sb3-contrib",
                "torch",
            ],
            description="Run pipelines/run_full_suite.py (includes ablation, audits, reports).",
            bootstrap=[
                "numpy>=1.23,<2.0",
                "wrapt>=1.11",
            ],
        ),
        "leadlag_hydra": StageDefinition(
            name="leadlag_hydra",
            requirements=[
                "-r",
                str(REPO_ROOT / "requirements-kaggle.txt"),
            ],
            entrypoint=ORCHESTRATOR_DIR / "stages" / "leadlag_hydra.py",
            uninstall=[],
            description="Execute lead-lag Hydra scenarios with optional multi-seed aggregation.",
            bootstrap=["numpy>=1.23,<2.0"],
        ),
    }

    parser = build_argument_parser(stage_registry)
    args = parser.parse_args()

    if args.list:
        list_stages(stage_registry)
        return

    selected_stages: list[StageDefinition]
    if args.stages:
        missing = [name for name in args.stages if name not in stage_registry]
        if missing:
            valid = ", ".join(stage_registry)
            raise SystemExit(f"Unknown stage(s): {missing}. Valid options: {valid}")
        selected_stages = [stage_registry[name] for name in args.stages]
    else:
        selected_stages = list(stage_registry.values())

    artifact_root = discover_artifact_root(args.artifacts_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    summary_path = artifact_root / "summary.json"

    venv_root = artifact_root.parent / ".stage_venvs"
    venv_root.mkdir(parents=True, exist_ok=True)

    created_envs: dict[str, Path] = {}

    for stage in selected_stages:
        stage_python, env_dir = create_virtualenv(stage.name, venv_root)
        created_envs[stage.name] = env_dir
        pip_install(stage_python, stage.requirements, bootstrap=stage.bootstrap)
        pip_check(stage_python)

        try:
            stdout_path, stderr_path, duration = run_stage(
                stage,
                artifact_root,
                stage_python,
            )
            status = "success"
            error = None
        except RuntimeError as exc:
            status = "failed"
            duration = 0.0
            stdout_path = (artifact_root / stage.name / "stdout.log").as_posix()
            stderr_path = (artifact_root / stage.name / "stderr.log").as_posix()
            error = str(exc)

        summary_record = {
            "stage": stage.name,
            "status": status,
            "duration_seconds": duration,
            "stdout_log": stdout_path,
            "stderr_log": stderr_path,
            "requirements": stage.requirements,
            "timestamp": time.time(),
            "description": stage.description,
        }
        if error:
            summary_record["error"] = error
            summary.append(summary_record)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            break

        summary.append(summary_record)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if not args.skip_cleanup:
            shutil.rmtree(env_dir, ignore_errors=True)

    if args.skip_cleanup:
        for record in summary:
            env_dir = created_envs.get(record["stage"])
            if env_dir:
                record["virtualenv"] = env_dir.as_posix()
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        shutil.rmtree(venv_root, ignore_errors=True)

    if not args.skip_zip:
        try:
            zip_path = package_artifacts(artifact_root)
            print(f"[orchestrator] archived artifacts to {zip_path}")
        except Exception as exc:
            print(f"[orchestrator] WARN: failed to create artifact archive: {exc}")


if __name__ == "__main__":
    main()
