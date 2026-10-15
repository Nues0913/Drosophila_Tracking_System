# syntax=docker/dockerfile:1
FROM osrf/ros:lyrical-desktop-full-resolute
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-lyrical-ros-gz ros-lyrical-foxglove-bridge python3-venv python3-colcon-common-extensions \
    python3-pip python3-pytest ros-lyrical-launch-testing-ament-cmake ros-lyrical-launch-testing-ros \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/flytrack-ros
ENV VIRTUAL_ENV=/opt/flytrack-ros PATH="/opt/flytrack-ros/bin:${PATH}" PYTHONUNBUFFERED=1
COPY docker/gz-camera-timing/ /opt/flytrack-camera-build/
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    python /opt/flytrack-camera-build/prepare.py --output /opt/flytrack-camera-build/sources && \
    . /opt/ros/lyrical/setup.sh && \
    cmake -S /opt/flytrack-camera-build -B /opt/flytrack-camera-build/build \
      -DCAMERA_SOURCE_DIR=/opt/flytrack-camera-build/sources \
      -DCMAKE_INSTALL_PREFIX=/opt/flytrack-camera-timing -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/flytrack-camera-build/build --parallel 2 && \
    cmake --install /opt/flytrack-camera-build/build && \
    rm -rf /opt/flytrack-camera-build/build
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY python/ python/
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.9.1
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" && \
    python -m pip install -e . 'PyYAML>=6,<7'
COPY ros2_ws/src/ ros2_ws/src/
WORKDIR /workspace/ros2_ws
RUN . /opt/ros/lyrical/setup.sh && python /usr/bin/colcon build --symlink-install \
    --cmake-args -DPython3_EXECUTABLE=/opt/flytrack-ros/bin/python
WORKDIR /workspace
COPY configs/ configs/
COPY scripts/ scripts/
COPY docker/ros-entrypoint.sh /flytrack-entrypoint.sh
ENV GZ_SIM_RESOURCE_PATH=/workspace/ros2_ws/install/flytrack_bringup/share/flytrack_bringup/models
ENTRYPOINT ["bash", "/flytrack-entrypoint.sh"]
CMD ["ros2", "launch", "flytrack_bringup", "world.launch.py"]
