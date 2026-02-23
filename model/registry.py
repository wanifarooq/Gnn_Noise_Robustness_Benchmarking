"""Trainer registry — maps method names to BaseTrainer subclasses."""

from model.base import BaseTrainer

TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {}

_discovered = False


def register(method_name: str):
    """Class decorator: ``@register('standard')`` on a BaseTrainer subclass."""

    def decorator(cls: type[BaseTrainer]) -> type[BaseTrainer]:
        if method_name in TRAINER_REGISTRY:
            raise ValueError(
                f"Duplicate registration for method '{method_name}': "
                f"{TRAINER_REGISTRY[method_name]} vs {cls}"
            )
        TRAINER_REGISTRY[method_name] = cls
        return cls

    return decorator


def discover_trainers() -> None:
    """Import every method module so their ``@register`` decorators execute.

    Scans ``model.methods`` via *pkgutil* — adding a new method is just
    dropping a file in that subpackage.  Safe to call multiple times.
    """
    global _discovered
    if _discovered:
        return
    import importlib
    import pkgutil
    import model.methods as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__):
        if not info.name.startswith('_'):
            importlib.import_module(f'model.methods.{info.name}')
    _discovered = True


def get_trainer(method_name: str, init_data: dict, config: dict) -> BaseTrainer:
    """Look up a registered trainer by *method_name* and instantiate it."""
    if method_name not in TRAINER_REGISTRY:
        registered = ', '.join(sorted(TRAINER_REGISTRY)) or '(none)'
        raise ValueError(
            f"Training method '{method_name}' not registered. "
            f"Available: [{registered}]"
        )
    return TRAINER_REGISTRY[method_name](init_data, config)
