"""Signal env -> gateway config bridge (``PlatformEntry.env_overrides``).

The signal-ts runtime is configured in config.yaml only (``platforms.signal.extra``); it has no
env credentials and no env allowlist. The one env value still honored is the home-channel mirror
``/sethome`` writes (``SIGNAL_HOME_CHANNEL``), applied only when Signal is configured in YAML.
"""

from gateway.config import Platform
from gateway.platform_env import home_step

ENV_ENABLE_CREDENTIALS = ()
ENV_OVERRIDES = (home_step(Platform.SIGNAL, "SIGNAL_HOME_CHANNEL"),)

# Authorization env names (``PlatformEntry`` fields). Access is config-owned
# (``extra.allow_from`` / ``extra.group_allow_from``, enforced by the adapter), so none are declared.
AUTHZ = {}
