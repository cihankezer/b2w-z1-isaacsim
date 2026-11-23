"""Render both T1 demo videos in one container (the second reuses the RTX shader cache).

    render_videos.py <targets.json> <out_dir> [name,...]   (default: fixed_targets,circle)
Runs evaluation/render_t1.py twice as child processes; commands are written to <out_dir>/commands.json.
"""
import json
import subprocess
import sys
from pathlib import Path

targets, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
script = str(Path(__file__).resolve().parents[1]/"render_t1.py")
jobs = [["--ids", ",".join(f"fixed-{i:02d}" for i in range(6)), "--name", "fixed_targets"],
        ["--ids", "circle-00-XY", "--name", "circle", "--trail"]]
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
jobs = [j for j in jobs if only is None or j[j.index("--name")+1] in only]
commands = [[sys.executable, script, "--targets", targets, "--out", str(out), *job, "--device", "cuda:0"] for job in jobs]
(out/f"commands{'_' + '_'.join(only) if only else ''}.json").write_text(json.dumps(commands, indent=1) + "\n")
for command, job in zip(commands, jobs):
    name = job[job.index("--name")+1]
    with (out/f"{name}.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    print(name, "exit", code, "video", (out/f"{name}.mp4").exists(), flush=True)
