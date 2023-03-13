import gin
import copy
import inspect

def get_all_gin_configurable_signatures():
    """Get the signatures for all things that are gin-configurable before gin has been applied.

    Returns
    -------
    A dictionary mapping, for each configurable, name (including full module path) to a dictionary mapping parameter
    name to default value. Example: If we only have the L1Loss configurable, we can expect

    .. code-block:: python

        {"torch.nn.L1Loss": {"size_average": None, "reduce": None, "reduction": "mean"}}

    as a result. Parameters might be :class:`inspect.Parameter.empty`.
    """
    gin_configurables = {}
    for name, selectable in gin.config._REGISTRY._selector_map.items():
        gin_configurables[name] = inspect.signature(selectable.wrapped)
    return gin_configurables


def get_all_gin_parameters():
    """Get parameters for all things that are gin-configurable after gin-settings have been applied.

    Returns
    -------
    A dictionary mapping, for each configurable, name (including full module path) to a dictionary mapping parameter
    name to default value. If multiple scopes are defined, we return multiple entries. This includes configurables that
    have not been set by gin.

    Example: If we have the L1Loss configurable and have it defined with different parameters in the scope
    `surface_head`, we can expect

    .. code-block:: python

    {
        ("", "torch.nn.L1Loss"):              {"size_average": None, "reduce": None, "reduction": "mean"},
        ("surface_head", "torch.nn.L1Loss"):  {"size_average": None, "reduce": None, "reduction": "sum"}
    }

    as a result. Parameters might be :class:`inspect.Parameter.empty`.

    Does not currently resolve macros or class definitions.
    """
    gin_signatures = get_all_gin_configurable_signatures()
    gin_configurables = {}
    empty_scope = ""
    for name, signature in gin_signatures.items():
        # Add the default parameters with an empty scope.
        gin_configurables[(empty_scope, name)] = {k: v.default for k, v in signature.parameters.items()}
    for (scope, name), param_dict in get_gin_set_params().items():
        gin_configurables[(scope, name)] = copy.deepcopy(gin_configurables[(empty_scope, name)])
        for param_name, new_val in param_dict.items():
            gin_configurables[(scope, name)][param_name] = new_val
    return gin_configurables


def get_gin_set_params():
    """Get parameters for all things that gin is modifying.

    Returns
    -------
    A dictionary mapping, for each configurable, a tuple of (scope, name) mapped to the parameter it is set to.
    scope is an empty string if this applies to everything.
    """
    return gin.config._CONFIG