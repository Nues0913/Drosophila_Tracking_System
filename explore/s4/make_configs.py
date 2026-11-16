"""Write the S4 BC and evaluation configs from the S2 ones (PM, #35). Host Python with PyYAML:
  python3 -I explore/s4/make_configs.py S2_CONFIG_DIR STD_FLOOR
S4 = S2 (formal VO-late BC + train.speed_gradient_balance target_ratio 0.3) + S3's desired-command labels
(env.teacher_accel_clip: false) + forward rows over all 134 bilateral DN types that train the core
(detach false), clipped at 5 sd, with std_floor and recalibration every round. Only those keys and the
output paths change. Evaluation: block 1,930,000,000 (unused by S2 1.86e9, M1 1.89e9, S3 1.90e9,
R2 1.91e9 and the S4 pre-check's environment seeds 1.92e9 + i).
"""
import sys
from pathlib import Path

import yaml

S2, floor = Path(sys.argv[1]), float(sys.argv[2])
here = Path(__file__).resolve().parent
types = (here / "dn_types.txt").read_text().split()
assert len(types) == 134 and "DNp09" in types, len(types)
for seed in (0, 1):
    config = yaml.safe_load((S2 / f"bc_c03_VO-late_seed_{seed}.yaml").read_text())
    assert config["train"]["speed_gradient_balance"] == {"target_ratio": 0.3, "max_scale": 10000.0}
    assert config["variant"] == "VO-late" and config["seed"] == seed
    assert "calibrate_forward_rows" not in config["train"] and "teacher_accel_clip" not in config["env"]
    config["output"] = f"runs/s4/bc/VO-late/seed_{seed}"
    config["env"]["teacher_accel_clip"] = False
    config["train"]["calibrate_forward_rows"] = {
        "types": types, "burn_in_steps": 60, "detach": False, "clip": 5.0, "std_floor": floor,
        "every_round": True,
    }
    header = (
        "# #35 S4 exploration (PM): the S2 c03 config of this chain (formal-v1 VO-late BC + speed/turn\n"
        "# gradient balance 0.3) plus S3's desired-command labels (env.teacher_accel_clip false) and forward\n"
        "# rows over the 134 bilateral DN types that train the core (calibrate_forward_rows). Written by\n"
        "# explore/s4/make_configs.py; only those keys and output changed.\n"
    )
    (here / "configs" / f"bc_s4_VO-late_seed_{seed}.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))
for name, template in (("baseline", "runs/bc/{variant}/seed_{seed}/actor"), ("s4", "runs/s4/bc/{variant}/seed_{seed}/actor")):
    config = yaml.safe_load((S2 / "eval_c03.yaml").read_text())
    assert config["evaluation"]["environment_seed"] == 1860000000
    config["evaluation"]["environment_seed"] = 1930000000
    config["evaluation"]["artifact_template"] = template
    config["output"] = "runs/s4/eval/" + ("baseline_VO-late" if name == "baseline" else "candidate_s4")
    header = (f"# #35 S4 exploration evaluation ({name}, VO-late only): configs/train/evaluate.yaml with the fresh\n"
              "# block environment_seed 1,930,000,000 and seeds 0 and 1. Written by explore/s4/make_configs.py.\n")
    (here / "configs" / f"eval_{name}.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))
print("ok")
