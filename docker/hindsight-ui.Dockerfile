# Build against vectorize-io/hindsight v0.9.0, commit
# b12646f49ec512136b9f709e608524ffed969668, as the build context.
# Only the control plane is rebuilt; the existing API and database stay put.
FROM node:26-bookworm-slim@sha256:9e6f9357d371591e32ab6f2d8a26d63bdd0d17c29eee3f4f3e7e454d9634bf73 AS build
WORKDIR /src
COPY package.json package-lock.json ./
COPY hindsight-clients/typescript ./hindsight-clients/typescript
COPY hindsight-control-plane ./hindsight-control-plane
RUN npm ci --ignore-scripts -w @vectorize-io/hindsight-client -w @vectorize-io/hindsight-control-plane
ENV NEXT_PUBLIC_BASE_PATH=/hindsight NEXT_TELEMETRY_DISABLED=1
RUN npm run build -w @vectorize-io/hindsight-control-plane

FROM node:26-bookworm-slim@sha256:9e6f9357d371591e32ab6f2d8a26d63bdd0d17c29eee3f4f3e7e454d9634bf73
WORKDIR /app
COPY --from=build --chown=node:node /src/hindsight-control-plane/standalone ./
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=9999 NEXT_PUBLIC_BASE_PATH=/hindsight NEXT_TELEMETRY_DISABLED=1
USER node
EXPOSE 9999
CMD ["node", "server.js"]
