from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
from functools import lru_cache
from mmengine.logging import print_log
from mmengine.registry import (
    build_from_cfg, build_runner_from_cfg, build_model_from_cfg, build_scheduler_from_cfg,
    MODELS, RUNNERS, RUNNER_CONSTRUCTORS, LOOPS, HOOKS, STRATEGIES, DATASETS, DATA_SAMPLERS,
    TRANSFORMS, MODEL_WRAPPERS, WEIGHT_INITIALIZERS, OPTIMIZERS, OPTIM_WRAPPERS,
    OPTIM_WRAPPER_CONSTRUCTORS, PARAM_SCHEDULERS, METRICS, EVALUATOR, TASK_UTILS,
    VISUALIZERS, VISBACKENDS, LOG_PROCESSORS, INFERENCERS, FUNCTIONS,
    Registry, init_default_scope, DefaultScope
)
from mmengine.registry import count_registered_modules, traverse_registry_tree
import torch.nn as nn

class RegistryError(Exception):
    """Base class for registry-related errors."""
    pass

class RegistryNotFoundError(RegistryError):
    """Raised when a registry is not found."""
    pass

class ModuleNotFoundError(RegistryError):
    """Raised when a module is not found in any registry."""
    pass

class RegistryManager:
    def __init__(self):
        self.registry_dict: Dict[str, Registry] = {
            'MODELS': MODELS,
            'RUNNERS': RUNNERS,
            'RUNNER_CONSTRUCTORS': RUNNER_CONSTRUCTORS,
            'LOOPS': LOOPS,
            'HOOKS': HOOKS,
            'STRATEGIES': STRATEGIES,
            'DATASETS': DATASETS,
            'DATA_SAMPLERS': DATA_SAMPLERS,
            'TRANSFORMS': TRANSFORMS,
            'MODEL_WRAPPERS': MODEL_WRAPPERS,
            'WEIGHT_INITIALIZERS': WEIGHT_INITIALIZERS,
            'OPTIMIZERS': OPTIMIZERS,
            'OPTIM_WRAPPERS': OPTIM_WRAPPERS,
            'OPTIM_WRAPPER_CONSTRUCTORS': OPTIM_WRAPPER_CONSTRUCTORS,
            'PARAM_SCHEDULERS': PARAM_SCHEDULERS,
            'METRICS': METRICS,
            'EVALUATOR': EVALUATOR,
            'TASK_UTILS': TASK_UTILS,
            'VISUALIZERS': VISUALIZERS,
            'VISBACKENDS': VISBACKENDS,
            'LOG_PROCESSORS': LOG_PROCESSORS,
            'INFERENCERS': INFERENCERS,
            'FUNCTIONS': FUNCTIONS
        }

    @lru_cache(maxsize=None)
    def get_registry(self, name: str) -> Registry:
        """Get a registry by name."""
        if name not in self.registry_dict:
            raise RegistryNotFoundError(f"Registry '{name}' not found.")
        return self.registry_dict[name]

    def create_registry(self, name: str, parent: Optional[str] = None, build_func: Optional[Callable] = None, 
                        scope: Optional[str] = None, locations: Optional[List[str]] = None) -> Registry:
        """Create a new registry or return existing one."""
        if name in self.registry_dict:
            print_log(f"Registry '{name}' already exists. Returning existing registry.", level='WARNING')
            return self.registry_dict[name]
        
        parent_registry = self.get_registry(parent) if parent else None
        new_registry = Registry(name, build_func=build_func, parent=parent_registry, 
                                scope=scope, locations=locations)
        self.registry_dict[name] = new_registry
        return new_registry

    def remove_registry(self, name: str) -> None:
        """Remove a registry by name."""
        if name in self.registry_dict:
            del self.registry_dict[name]
            print_log(f"Registry '{name}' has been removed.", level='INFO')
        else:
            print_log(f"Registry '{name}' not found, cannot remove.", level='WARNING')

    def list_registries(self) -> List[str]:
        """List all registered registries."""
        return list(self.registry_dict.keys())

    def clear_registries(self) -> None:
        """Clear all registries."""
        self.registry_dict.clear()
        print_log("All registries have been cleared.", level='INFO')

    def build(self, cfg: Dict[str, Any], registry_name: str, *args: Any, **kwargs: Any) -> Any:
        """Build an instance from config dict."""
        registry = self.get_registry(registry_name)
        scope = cfg.pop('_scope_', None)
        if scope:
            with self.switch_scope_and_registry(scope):
                return registry.build(cfg, *args, **kwargs)
        return registry.build(cfg, *args, **kwargs)

    def build_from_cfg(self, cfg: Dict[str, Any], registry: Registry, default_args: Optional[Dict[str, Any]] = None) -> Any:
        """Build a module from config dict."""
        return build_from_cfg(cfg, registry, default_args)

    def build_runner(self, cfg: Dict[str, Any]) -> Any:
        """Build a runner from config dict."""
        return build_runner_from_cfg(cfg, self.get_registry('RUNNERS'))

    def build_model(self, cfg: Union[Dict[str, Any], List[Dict[str, Any]]], default_args: Optional[Dict[str, Any]] = None) -> Any:
        """Build a model from config dict or list of config dicts."""
        if isinstance(cfg, list):
            models = [build_model_from_cfg(c, self.get_registry('MODELS'), default_args) for c in cfg]
            return nn.Sequential(*models)
        return build_model_from_cfg(cfg, self.get_registry('MODELS'), default_args)

    def build_scheduler(self, cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
        """Build a parameter scheduler from config dict."""
        if 'convert_to_iter_based' in cfg:
            if default_args is None or 'epoch_length' not in default_args:
                raise ValueError("'epoch_length' must be provided in default_args when using 'convert_to_iter_based'")
            cfg = cfg.copy()
            cfg['convert_to_iter_based'] = True
        return build_scheduler_from_cfg(cfg, self.get_registry('PARAM_SCHEDULERS'), default_args)

    def register_module(self, registry_name: str, name: Optional[str] = None, 
                        force: bool = False, module: Optional[Union[Type, Callable]] = None) -> Callable:
        """Register a module."""
        registry = self.get_registry(registry_name)
        return registry.register_module(name=name, force=force, module=module)

    def init_default_scope(self, scope: str) -> None:
        """Initialize the default scope."""
        init_default_scope(scope)
        print_log(f"Default scope initialized to '{scope}'", level='INFO')

    def get_current_scope(self) -> Optional[str]:
        """Get the current default scope."""
        return DefaultScope.get_current_instance().scope_name if DefaultScope.get_current_instance() else None

    def overwrite_default_scope(self, scope: str) -> None:
        """Overwrite the default scope."""
        DefaultScope.overwrite_default_scope(scope)
        print_log(f"Default scope overwritten to '{scope}'", level='INFO')

    def count_registered_modules(self, save_path: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        """Count registered modules."""
        return count_registered_modules(save_path, verbose)

    def traverse_registry_tree(self, registry_name: str, verbose: bool = False) -> List[Dict[str, Any]]:
        """Traverse the registry tree."""
        registry = self.get_registry(registry_name)
        return traverse_registry_tree(registry, verbose)

    @contextmanager
    def switch_scope_and_registry(self, scope: str):
        """Temporarily switch the default scope."""
        original_scope = self.get_current_scope()
        self.init_default_scope(scope)
        try:
            yield
        finally:
            self.init_default_scope(original_scope)

    def switch_scope_and_registry_method(self, scope: str) -> Registry:
        """Switch the default scope and return the corresponding registry."""
        self.init_default_scope(scope)
        return self.get_registry(scope)

    def import_from_location(self, registry_name: str) -> None:
        """Import modules from predefined locations for a registry."""
        registry = self.get_registry(registry_name)
        registry.import_from_location()

    def split_scope_key(self, registry_name: str, key: str) -> tuple:
        """Split scope and key."""
        registry = self.get_registry(registry_name)
        return registry.split_scope_key(key)

    def get(self, registry_name: str, key: str) -> Any:
        """Get an item from a registry."""
        registry = self.get_registry(registry_name)
        return registry.get(key)

    def infer_scope(self, registry_name: str) -> str:
        """Infer the scope of a registry."""
        registry = self.get_registry(registry_name)
        return registry.infer_scope()

    def add_registry(self, name: str, registry: Registry, force: bool = False) -> None:
        """Add a new registry dynamically."""
        if name in self.registry_dict and not force:
            raise ValueError(f"Registry '{name}' already exists. Use force=True to overwrite.")
        self.registry_dict[name] = registry
        print_log(f"Registry '{name}' added successfully.", level='INFO')

    def clone_registry(self, source_name: str, target_name: str) -> None:
        """Clone a registry."""
        source_registry = self.get_registry(source_name)
        new_registry = Registry(target_name, build_func=source_registry._build_func,
                                parent=source_registry.parent, scope=source_registry._scope,
                                locations=source_registry._locations)
        new_registry._module_dict = source_registry._module_dict.copy()
        self.add_registry(target_name, new_registry)

    def is_module_registered(self, module: Union[str, Type, Callable]) -> bool:
        """Check if a module is registered in any registry."""
        for registry in self.registry_dict.values():
            if isinstance(module, str):
                if module in registry._module_dict:
                    return True
            else:
                if module in registry._module_dict.values():
                    return True
        return False

    def get_all_registered_modules(self) -> Dict[str, List[str]]:
        """Get all registered modules across all registries."""
        all_modules = {}
        for registry_name, registry in self.registry_dict.items():
            all_modules[registry_name] = list(registry._module_dict.keys())
        return all_modules

    def compare_registries(self, registry1_name: str, registry2_name: str) -> Dict[str, List[str]]:
        """Compare two registries and find differences."""
        registry1 = self.get_registry(registry1_name)
        registry2 = self.get_registry(registry2_name)
        modules1 = set(registry1._module_dict.keys())
        modules2 = set(registry2._module_dict.keys())
        return {
            'only_in_' + registry1_name: list(modules1 - modules2),
            'only_in_' + registry2_name: list(modules2 - modules1),
            'in_both': list(modules1 & modules2)
        }