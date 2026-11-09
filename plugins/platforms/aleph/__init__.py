# Absolute import (not ``from .register``): the plugin loader executes this package under a synthetic
# name, and the adapter's absolute sibling imports would otherwise load a second copy of each module.
from plugins.platforms.aleph.register import register

__all__ = ["register"]
