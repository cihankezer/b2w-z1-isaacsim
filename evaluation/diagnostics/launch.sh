#!/usr/bin/env bash
# Run one T1 diagnostic in a uniquely named, read-only-repo Isaac container.
#   launch.sh cpu|gpu <label> <timeout_s> <script-relative-to-repo> [args...]
# Inside the container the run directory is /t1-run. Outcome -> <run>/host_result.json.
# On timeout only this run's container is stopped and the outcome is "hang".
set -uo pipefail
MODE=$1; LABEL=$2; LIMIT=$3; SCRIPT=$4; shift 4
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE_TAG=isaac-lab:2.3.2
IMAGE_ID=$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}') || { echo "missing image $IMAGE_TAG"; exit 2; }
GPU_UUID=${ISAAC_GPU_UUID:-$(awk '/uuid:/{print $2; exit}' "$ROOT/config.yaml")}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')
RUN="$ROOT/repos/robot_lab/logs/t1/diag-$LABEL-$STAMP"
NAME="b2w-t1-diag-$LABEL-$STAMP"
mkdir -p "$RUN"/{artifacts,cache/{main,computecache,logs,config,data,pkg}}
chmod -R 777 "$RUN"  # this new run directory only; the container runs as UID 1234
CMD=(docker run --rm --name "$NAME" --shm-size=2g --stop-timeout 10 --entrypoint /isaac-sim/python.sh
     --mount "type=bind,src=$ROOT,dst=/workspace,readonly" --mount "type=bind,src=$RUN,dst=/t1-run"
     -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e T1_IMAGE_ID="$IMAGE_ID" -e T1_IMAGE_TAG="$IMAGE_TAG")
# T1 robot asset (container path under /workspace); unset -> config default robots_usd/b2w_z1_gripper.usd
[ -n "${B2W_Z1_GRIPPER_USD:-}" ] && CMD+=(-e B2W_Z1_GRIPPER_USD="$B2W_Z1_GRIPPER_USD")
[ -n "${B2W_Z1_USD:-}" ] && CMD+=(-e B2W_Z1_USD="$B2W_Z1_USD")
if [ "$MODE" = gpu ]; then
  [ -n "$GPU_UUID" ] && [ "$GPU_UUID" != all ] || { echo "no allocated GPU UUID"; exit 2; }
  CMD+=(--gpus "device=$GPU_UUID" --network bridge -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES
        -e NVIDIA_DRIVER_CAPABILITIES=all -e ISAAC_GPU_UUID="$GPU_UUID"
        --mount "type=bind,src=$RUN/cache/main,dst=/isaac-sim/.cache"
        --mount "type=bind,src=$RUN/cache/computecache,dst=/isaac-sim/.nv/ComputeCache"
        --mount "type=bind,src=$RUN/cache/logs,dst=/isaac-sim/.nvidia-omniverse/logs"
        --mount "type=bind,src=$RUN/cache/config,dst=/isaac-sim/.nvidia-omniverse/config"
        --mount "type=bind,src=$RUN/cache/data,dst=/isaac-sim/.local/share/ov/data"
        --mount "type=bind,src=$RUN/cache/pkg,dst=/isaac-sim/.local/share/ov/pkg"
        -w /workspace/repos/robot_lab)
else
  CMD+=(--network none -w /workspace)
fi
CMD+=("$IMAGE_ID" "/workspace/$SCRIPT" "$@")
printf '%s\n' "${CMD[@]}" > "$RUN/launch.txt"
# Source snapshot of everything the run can import from this repo, plus manifest with hashes.
TASK_DIR=training/robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/manipulation/standing_ee
mkdir -p "$RUN/source"
( cd "$ROOT" && printf '%s\n' "$SCRIPT" evaluation/diagnostics/*.py evaluation/diagnostics/launch.sh \
    evaluation/eval_t1.py evaluation/diagnose_t1_standing.py $TASK_DIR/*.py tests/test_t1*.py docs/protocols/TASK_T1.md \
  | sort -u | xargs cp --parents -t "$RUN/source/" )
[ "$MODE" = gpu ] || GPU_UUID=none
python3 - "$RUN" "$ROOT" "$SCRIPT" "$MODE" "$GPU_UUID" "$IMAGE_ID" "$IMAGE_TAG" "$NAME" <<'EOF'
import hashlib, json, pathlib, subprocess, sys
run, root, script, mode, gpu, image_id, image_tag, name = sys.argv[1:]
src = pathlib.Path(run) / "source"
git = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True).stdout
json.dump(dict(script=script, mode=mode, container=name, gpu_uuid=gpu, image_tag=image_tag, image_id=image_id,
               command=open(f"{run}/launch.txt").read().splitlines(),
               git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--short").splitlines(),
               files={str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(src.rglob("*")) if p.is_file()}),
          open(f"{run}/manifest.json", "w"), indent=1)
EOF
START=$(date +%s.%N)
timeout --signal=TERM "$LIMIT" "${CMD[@]}" > "$RUN/stdout.log" 2>&1
CODE=$?
chmod -R a+rX "$RUN/source"
OUTCOME=exited
if [ $CODE -eq 124 ]; then
  OUTCOME=hang
  docker stop -t 5 "$NAME" > /dev/null 2>&1; docker rm -f "$NAME" > /dev/null 2>&1
fi
END=$(date +%s.%N)
printf '{"outcome":"%s","exit_code":%d,"wall_s":%.1f,"container":"%s","gpu_uuid":"%s","image_id":"%s"}\n' \
  "$OUTCOME" "$CODE" "$(echo "$END - $START" | bc)" "$NAME" "${GPU_UUID}" "$IMAGE_ID" > "$RUN/host_result.json"
echo "$RUN"
