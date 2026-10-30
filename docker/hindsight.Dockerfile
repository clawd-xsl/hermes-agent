# Merge the already-tested subpath UI into the same upstream API version.
# Build the UI donor with docker/hindsight-ui.Dockerfile first when necessary.
ARG HINDSIGHT_UI_IMAGE=hermes-hindsight-ui:1.1.0-hindsight
ARG HINDSIGHT_BASE_IMAGE=ghcr.io/vectorize-io/hindsight:1.1.0-slim@sha256:fbc37d194bce007b6cce2fd9f66cd7f8c050bd18b80e662090a92835d9cb191d
FROM ${HINDSIGHT_UI_IMAGE} AS ui
FROM ${HINDSIGHT_BASE_IMAGE}
USER root

# Replace, rather than overlay, the old root-path bundle so no obsolete routes
# or static chunks survive. This path is image code, not a mounted data volume.
RUN rm -rf /app/control-plane
COPY --from=ui --chown=hindsight:hindsight --chmod=755 /app/ /app/control-plane/
# Keep upstream's Node runtime and process lifecycle; only the UI bundle moves.
ENV NEXT_PUBLIC_BASE_PATH=/hindsight NEXT_TELEMETRY_DISABLED=1
USER hindsight
# The slim upstream runtime no longer includes curl.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=20 \
    CMD ["/app/api/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8888/health', timeout=3)"]
