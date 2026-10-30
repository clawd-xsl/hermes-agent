# Hindsight API and UI in one container

`docker/hindsight.Dockerfile` layers the same-version portal UI onto the pinned
upstream Hindsight image. No upstream source fork or custom process manager is
required: `/app/start-all.sh` already manages the API and control plane together.
The API code, Python environment, Node runtime and database schema stay unchanged.

Build the UI donor from the pinned upstream checkout as documented in
`docker/hindsight-ui.Dockerfile`, then build the combined image from this checkout:

```sh
docker build -f docker/hindsight.Dockerfile -t hermes-hindsight:0.9.0-portal .
```

Use the combined image for the existing `hindsight` Compose service, preserving
its env file, database settings, OAuth mount, worker identity and network aliases.
Set `HINDSIGHT_CP_DATAPLANE_API_URL=http://127.0.0.1:8888` and
`NEXT_PUBLIC_BASE_PATH=/hindsight`. Publish `127.0.0.1:9998:9999` so the Cola portal
keeps its existing upstream. Remove the separate `hindsight-ui` service.

Before production cutover, start a disposable UI-only instance with
`HINDSIGHT_ENABLE_API=false`, point its dataplane at the running API, and verify
authenticated dashboard, static assets and bank-list calls. It needs no database
or OAuth volume mount. After cutover, check API health, unchanged Alembic revision,
memory/document counts, portal auto-login and static assets. A recall embedding
authentication failure is a separate provider problem, not fixed by this merge.

For a future version upgrade, build both API and UI at the same version, test
schema migration and Hermes compatibility separately, and do not treat an image
rollback as a database rollback.
