from src.models.pytorch.transformer_base import TransformerClassifierTorch

class KobertClassifierTorch(TransformerClassifierTorch):
    def __init__(self, config):
        cfg = config.copy()
        cfg['pretrained_model_name'] = cfg.get('pretrained_model_name', 'skt/kobert-base-v1')
        super().__init__(cfg)
