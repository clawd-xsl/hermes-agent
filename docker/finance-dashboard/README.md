# Finance inside Hermes

This deployment-only overlay adds the user's existing Plaid Ledger app to
Hermes's s6 service tree. It does not change the gateway or install a second
copy of the app. The app, locked virtualenv and financial state remain under
the persistent `HERMES_HOME` mount.

Build on top of the currently deployed Hermes runtime:

```sh
docker build -f docker/finance-dashboard.Dockerfile \
  --build-arg HERMES_BASE_IMAGE=hermes:claude-owned-compaction-20261026 \
  -t hermes:finance-supervised-20261029 .
```

Use that image for the main Hermes service, with `DASHBOARD_HOST=0.0.0.0`,
`DASHBOARD_PORT=8766` and `DASHBOARD_BASE_PATH=/finance`. Publish
`127.0.0.1:8766:8766` only; retain the Cola portal's existing `/finance/` proxy.
Remove the old standalone finance service rather than running two writers.

Inside Hermes, the normal agent user can run:

```sh
finance-dashboard status
finance-dashboard stop
finance-dashboard start
finance-dashboard restart
finance-dashboard health
finance-dashboard logs
```

The service starts on container boot and s6 restarts it on process exit.
App logs rotate at 2 MB with three backups under `$HERMES_HOME/logs`.
Health checks prove HTTP availability, not successful Plaid refresh: inspect
only the `live` status in `/api/overview` to check refresh without disclosing
account data. Credentials supplied via Compose `env_file: format: raw` must
not include dotenv-style wrapping quotes.

Run the isolated bootstrap tests with
`scripts/run_tests.sh docker/finance-dashboard/tests/test_launcher.py -q`.
Also verify the built image as the runtime UID: source-directory modes must
not make `/etc`, `/usr`, `/opt` or the Python launcher root-only.
