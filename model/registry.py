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
    """Import every model module so their ``@register`` decorators execute.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _discovered
    if _discovered:
        return
    import model.Standard            # noqa: F401
    import model.Positive_Eigenvalues  # noqa: F401
    import model.GCOD_loss           # noqa: F401
    import model.NRGNN               # noqa: F401
    import model.PI_GNN              # noqa: F401
    import model.CR_GNN              # noqa: F401
    import model.CommunityDefense    # noqa: F401
    import model.RTGNN               # noqa: F401
    import model.GraphCleaner        # noqa: F401
    import model.UnionNET            # noqa: F401
    import model.GNN_Cleaner         # noqa: F401
    import model.ERASE               # noqa: F401
    import model.GNNGuard            # noqa: F401
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
