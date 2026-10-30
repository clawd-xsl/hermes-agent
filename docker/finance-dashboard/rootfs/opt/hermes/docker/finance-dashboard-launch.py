"""Run the user-owned finance app with bounded, agent-readable logs."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def main():
    import uvicorn

    log_dir = Path(os.environ["HERMES_HOME"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[RotatingFileHandler(log_dir / "finance-dashboard.log", maxBytes=2_000_000, backupCount=3)],
        force=True,
    )
    sys.path.insert(0, os.getcwd())
    uvicorn.run(
        "app:app", host=os.environ.get("DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("DASHBOARD_PORT", "8766")), log_config=None,
    )


if __name__ == "__main__":
    main()
