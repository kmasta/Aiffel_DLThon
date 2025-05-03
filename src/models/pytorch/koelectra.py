from src.models.pytorch.transformer_base import TransformerClassifierTorch

class KoelectraClassifierTorch(TransformerClassifierTorch):
    def __init__(self, config):
        cfg = config.copy()
        cfg['pretrained_model_name'] = cfg.get('pretrained_model_name', 'monologg/koelectra-base-v3-discriminator')
        super().__init__(cfg)
