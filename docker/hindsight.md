# Hindsight API and UI in one container

`docker/hindsight.Dockerfile` layers the same-version portal UI onto the pinned
upstream Hindsight image. No upstream source fork or custom process manager is
required: `/app/start-all.sh` already manages the API and control plane together.
The overlay does not patch API code, Python dependencies or the Node runtime.
Database schema changes come only from the selected upstream release's migrations.

Build the UI donor from the pinned upstream checkout as documented in
`docker/hindsight-ui.Dockerfile`, then build the combined image from this checkout:

```sh
docker build -f docker/hindsight.Dockerfile -t hermes-hindsight:1.1.0-portal .
```

Use the combined image for the existing `hindsight` Compose service, preserving
its env file, database settings, OAuth mount, worker identity and network aliases.
Set `HINDSIGHT_CP_DATAPLANE_API_URL=http://127.0.0.1:8888` and
`NEXT_PUBLIC_BASE_PATH=/hindsight`. Publish `127.0.0.1:9998:9999` so the Cola portal
keeps its existing upstream. Remove the separate `hindsight-ui` service.
The 1.1.0 slim image has no curl: use the image's Python/urllib healthcheck;
do not override it in Compose with the retired curl-based probe.

Before production cutover, start a disposable UI-only instance with
`HINDSIGHT_ENABLE_API=false`, point its dataplane at the running API, and verify
authenticated dashboard, static assets and bank-list calls. It needs no database
or OAuth volume mount. After cutover, check API health, unchanged Alembic revision,
memory/document counts, portal auto-login and static assets. A recall embedding
authentication failure is a separate provider problem, not fixed by this merge.

## Version upgrades

Build both API and UI at the same version. The defaults pin upstream 1.1.0 and
the matching UI donor. Run migrations against an isolated restored database first,
with the same vector dimension (384 here). Keep the staging worker disabled so it
cannot replay production jobs or send notifications. Use separate OAuth credentials
for tests that may refresh tokens; a read-only mount does not prevent server-side
refresh-token rotation.

The 0.9.0-to-1.1.0 rehearsal preserved all 15,205 memory vectors and 250 documents,
advanced Alembic from `b3e8d1c6f4a9` to `c7e2f9a4d1b8`, and decoded bank lists,
bank config and document lists through Hermes's existing hindsight-client 0.6.1.
The pre-existing OAuth embeddings 401 remains a separate provider issue.

Production 1.1.0 was validated through the Cola portal and Hermes's existing SDK.
The final pre-upgrade database restore point is kept locally at
`/home/shanlix/Hermes/services/postgres/backups/hindsight-pre-1.1.0.dump` (0600).
It contains private memory data: never commit, upload or include it in diagnostics.

For production, stop only Hindsight, take a final local private database dump,
run the already-tested migration, then start the new image against the same
database and OAuth mount. Validate API/UI health, auto-login, read APIs and vector
counts. Do not restart Hermes or change its SDK/session merely to upgrade the server.

Do not treat an image rollback as a database rollback. A rollback after writes
resume must account for those new writes; retain the pre-upgrade restore point
until the new version has been accepted.
