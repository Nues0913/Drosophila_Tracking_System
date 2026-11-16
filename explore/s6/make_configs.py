"""Write the S6 BC and evaluation configs from the S4 ones (PM, #35). Host Python with PyYAML:
  python3 -I explore/s6/make_configs.py
S6 = formal VO-late BC + S2's speed/turn gradient balance 0.3 + S3's desired-command labels (the S4 recipe
WITHOUT its DN-population forward rows: the speed is read from DNp09 alone, as in formal-v1), trained on the
derived graph runs/s6/graph/graph.npz (explore/s6/derive_graph.py: the formal subgraph plus the LC9 cells and
their real LC9 -> DNp09 edges) from the formal pretrain actor transplanted onto that graph
(explore/s6/transplant.py). Only graph_path, init_artifact, output and the removed rows change.
Evaluation: block 1,980,000,000 (unused: S2 1.86e9, M1 1.88e9/1.89e9, S3 1.90e9, R2 1.91e9, S4 1.92e9/1.93e9,
oracle 1.94e9, M2 1.95e9, probes 1.96e9/1.97e9).
"""
from pathlib import Path

import yaml

here = Path(__file__).resolve().parent
S4 = here.parent / "s4" / "configs"
for seed in (0, 1):
    config = yaml.safe_load((S4 / f"bc_s4_VO-late_seed_{seed}.yaml").read_text())
    assert config["train"]["speed_gradient_balance"] == {"target_ratio": 0.3, "max_scale": 10000.0}
    assert config["env"]["teacher_accel_clip"] is False and config["seed"] == seed
    assert config["init_artifact"] == f"runs/pretrain/VO-late/seed_{seed}/odor/actor"
    del config["train"]["calibrate_forward_rows"]
    config["graph_path"] = "runs/s6/graph/graph.npz"
    config["init_artifact"] = f"runs/s6/init/VO-late/seed_{seed}/actor"
    config["output"] = f"runs/s6/bc/VO-late/seed_{seed}"
    header = (
        "# #35 S6 exploration (PM): formal-v1 VO-late BC + speed/turn gradient balance 0.3 + desired-command\n"
        "# labels, on the derived graph that gives DNp09 its real LC9 input back; speed read from DNp09 alone.\n"
        "# Written by explore/s6/make_configs.py from the S4 config: graph_path, init_artifact and output\n"
        "# changed, calibrate_forward_rows removed.\n"
    )
    (here / "configs" / f"bc_s6_VO-late_seed_{seed}.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))
for name, template in (("baseline", "runs/bc/{variant}/seed_{seed}/actor"), ("s6", "runs/s6/bc/{variant}/seed_{seed}/actor")):
    config = yaml.safe_load((S4 / "eval_s4.yaml").read_text())
    assert config["evaluation"]["environment_seed"] == 1930000000
    config["evaluation"]["environment_seed"] = 1980000000
    config["evaluation"]["artifact_template"] = template
    config["output"] = "runs/s6/eval/" + ("baseline_VO-late" if name == "baseline" else "candidate_s6")
    header = (f"# #35 S6 exploration evaluation ({name}, VO-late only): configs/train/evaluate.yaml with the fresh\n"
              "# block environment_seed 1,980,000,000 and seeds 0 and 1. Written by explore/s6/make_configs.py.\n")
    (here / "configs" / f"eval_{name}.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))
print("ok")
