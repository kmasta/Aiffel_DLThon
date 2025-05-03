from src.models.pytorch.transformer_base import TransformerClassifierTorch

class KcelectraClassifierTorch(TransformerClassifierTorch):
    def __init__(self, config):
        cfg = config.copy()
        cfg['pretrained_model_name'] = cfg.get('pretrained_model_name', 'beomi/KcELECTRA-base')
        super().__init__(cfg)
