# Deployment overlay: keep the gateway/runtime image unchanged underneath.
ARG HERMES_BASE_IMAGE=hermes:claude-owned-compaction-20261026
FROM ${HERMES_BASE_IMAGE}
USER root
COPY --chmod=755 docker/finance-dashboard/rootfs/ /
RUN chmod 0644 /opt/hermes/docker/finance-dashboard-launch.py \
      /etc/s6-overlay/s6-rc.d/finance-dashboard/type
