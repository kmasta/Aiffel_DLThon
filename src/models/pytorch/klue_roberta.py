from src.models.pytorch.transformer_base import TransformerClassifierTorch

class Klue_robertaClassifierTorch(TransformerClassifierTorch):
    def __init__(self, config):
        cfg = config.copy()
        cfg['pretrained_model_name'] = cfg.get('pretrained_model_name', 'klue/roberta-base')
        super().__init__(cfg)
