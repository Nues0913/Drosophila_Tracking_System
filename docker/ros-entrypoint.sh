#!/usr/bin/env bash
set -e
source /opt/ros/lyrical/setup.bash
source /workspace/ros2_ws/install/setup.bash
export PATH="/opt/flytrack-ros/bin:$PATH"
export FLYTRACK_CLOCK_DOMAIN="$(python -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("/proc/sys/kernel/random/boot_id").read_bytes()).hexdigest())')"
export LD_LIBRARY_PATH="/opt/flytrack-camera-timing/lib:${LD_LIBRARY_PATH:-}"
exec "$@"
