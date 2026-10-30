# Merge the already-tested subpath UI into the same upstream API version.
# Build the UI donor with docker/hindsight-ui.Dockerfile first when necessary.
ARG HINDSIGHT_UI_IMAGE=hermes-hindsight-ui:0.9.0-hindsight
ARG HINDSIGHT_BASE_IMAGE=ghcr.io/vectorize-io/hindsight:0.9.0-slim@sha256:3ddd4b054aae6763b402f037ba89201e71ab0e0f48f536df89e9767122c656a0
FROM ${HINDSIGHT_UI_IMAGE} AS ui
FROM ${HINDSIGHT_BASE_IMAGE}

# Replace, rather than overlay, the old root-path bundle so no obsolete routes
# or static chunks survive. This path is image code, not a mounted data volume.
RUN rm -rf /app/control-plane
COPY --from=ui --chmod=755 /app/ /app/control-plane/
# Keep upstream's Node 20 runtime and process lifecycle; only the UI bundle moves.
ENV NEXT_PUBLIC_BASE_PATH=/hindsight NEXT_TELEMETRY_DISABLED=1
