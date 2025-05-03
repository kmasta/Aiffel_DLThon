import importlib

def load_model(config):
    fw = config['framework']
    name = config['model_name']
    module = importlib.import_module(f"src.models.{fw}.{name}")
    cls = getattr(module, f"{name.capitalize()}Classifier{'Torch' if fw=='pytorch' else 'Keras'}")
    return cls(config)
