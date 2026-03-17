"""Method helper registry — maps method names to MethodHelper subclasses.

Coexists with the old model.registry during the migration period.  Once all
methods are migrated to helpers, the old registry can be removed.
"""

import importlib
import pkgutil
from methods.base_helper import MethodHelper

HELPER_REGISTRY: dict[str, type[MethodHelper]] = {}

_discovered = False


def register_helper(method_name: str):
    """Class decorator: ``@register_helper('standard')`` on a MethodHelper subclass."""

    def decorator(cls: type[MethodHelper]) -> type[MethodHelper]:
        if method_name in HELPER_REGISTRY:
            raise ValueError(
                f"Duplicate helper registration for method '{method_name}': "
                f"{HELPER_REGISTRY[method_name]} vs {cls}"
            )
        HELPER_REGISTRY[method_name] = cls
        return cls

    return decorator


def discover_helpers() -> None:
    """Import every helper module so their ``@register_helper`` decorators execute.

    Scans ``methods`` package via pkgutil — adding a new helper is just
    dropping a file in that package.  Safe to call multiple times.
    """
    global _discovered
    if _discovered:
        return
    import methods as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__):
        if not info.name.startswith('_') and info.name not in ('base_helper', 'registry'):
            try:
                importlib.import_module(f'methods.{info.name}')
            except ImportError:
                pass
    _discovered = True


def get_helper(method_name: str) -> MethodHelper | None:
    """Look up a registered helper by method_name.

    Returns None if no helper is registered (falls back to old trainer system).
    """
    discover_helpers()
    cls = HELPER_REGISTRY.get(method_name)
    if cls is None:
        return None
    return cls()


def has_helper(method_name: str) -> bool:
    """Check if a helper is registered for the given method."""
    discover_helpers()
    return method_name in HELPER_REGISTRY
