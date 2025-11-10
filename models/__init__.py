import importlib


def load_model(model_name):
    """
    Dynamically load a model from the models package.

    Args:
        model_name (str): The name of the model file without `.py` extension.

    Returns:
        Model class, Collator class
    """
    module_name = f"models.{model_name}"

    try:
        module = importlib.import_module(module_name)
        return module.Model, module.Collator  # Ensure these exist in your module
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}")
    except AttributeError:
        raise AttributeError(f"Module '{module_name}' does not contain 'Model' or 'Collator'.")
