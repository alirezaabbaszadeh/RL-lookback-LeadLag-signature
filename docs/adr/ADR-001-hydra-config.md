# ADR-001: Hydra-based Configuration System

## Status
Accepted – 2025-10-11

## Context
The project requires a flexible and reproducible way to configure experiments. Prior to this decision each runner (`run_scenario`, `run_dynamic`, `run_rl`) consumed a static YAML file with limited ability to override parameters such as seeds or output directories. Running multiple scenarios or multi-seed sweeps required manual scripting.

## Decision
Adopt [Hydra](https://hydra.cc) to manage configuration. The entry point `hydra_main.py` loads `configs/config.yaml`, supports scenario defaults, multi-seed runs, and sequential execution of multiple scenarios. Scenario descriptors live under `configs/scenario/` and can be referenced by name or overridden inline.

## Consequences
- Pros: Dynamic CLI overrides, scenario composition, simple multi-seed orchestration, and systematic logging of outputs.
- Cons: Requires extra dependency (`hydra-core`) and learning curve for contributors unfamiliar with Hydra. Tests and CI must ensure the new interface remains stable.

## Links
- `hydra_main.py`
- `training/runner_multiseed.py`
- `docs/config_reference.md`
