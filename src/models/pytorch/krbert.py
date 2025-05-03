from src.models.pytorch.transformer_base import TransformerClassifierTorch

class KrbertClassifierTorch(TransformerClassifierTorch):
    def __init__(self, config):
        cfg = config.copy()
        cfg['pretrained_model_name'] = cfg.get(
            'pretrained_model_name', 'snunlp/KR-BERT-char16424'
        )
        super().__init__(cfg)
