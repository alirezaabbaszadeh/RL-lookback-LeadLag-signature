param(
  [string]$Scenario = "fixed_30",
  [string]$OutputRoot = "results"
)

python hydra_main.py --scenario $Scenario --output_root $OutputRoot

