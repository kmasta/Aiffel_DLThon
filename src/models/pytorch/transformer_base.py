import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from src.models.base import BaseModel
from src.metrics import compute_metrics
from transformers.trainer_callback import TrainerCallback


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False    # ← 여기서 token_type_ids 생성 멈춤
        )
        self.labels = torch.tensor(labels)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        # ─ token_type_ids 가 있으면 제거합니다 (안 생성된 경우 pop 은 무시됨)
        item = {k: v[idx] for k, v in self.enc.items()}
        item.pop('token_type_ids', None)  
        item['labels'] = self.labels[idx]
        return item

class TransformerClassifierTorch(BaseModel):
    def __init__(self, config):
        super().__init__() 
        self.config = config
        self.load_state(config['pretrained_model_name'])

    def train_model(self, train_texts, train_labels, val_texts, val_labels, config, log_callback=False):
        from transformers import TrainingArguments, Trainer
        import os
        # Prepare datasets
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, config['max_length'])
        eval_dataset = TextDataset(val_texts, val_labels, self.tokenizer, config['max_length'])

        # Define training arguments
        args = TrainingArguments(
            output_dir=os.path.join('checkpoints', config.get('experiment_name', 'model')),
            num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            eval_strategy='epoch',
            logging_dir=os.path.join('logs', config.get('experiment_name', 'model')),
            #logging_strategy='steps',
            #logging_steps=config.get('logging_steps', 100),
            optim      = config.get("optimizer", "adamw_torch"),           # Optimizer 종류
            optim_args = config.get("optim_args", None),               # Optimizer 초기화 인자
            learning_rate   = float(config['lr']),                # 초기 학습률
            lr_scheduler_type = config.get("lr_scheduler_type", "linear"),  
            warmup_steps      = config.get("warmup_steps", 0),      
            warmup_ratio      = config.get("warmup_ratio", 0.0),
            evaluation_strategy      = "epoch",
            load_best_model_at_end   = True,
            metric_for_best_model    = "f1",
            greater_is_better        = True,    
            save_strategy='epoch',
            seed=config.get('seed', 42),
            report_to=[],
            remove_unused_columns=False
        )
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda p: compute_metrics(p.label_ids, p.predictions.argmax(-1)),
            callbacks         = [EarlyStoppingCallback(early_stopping_patience= config.get("patience", 3))]
        )

        # Train the model
        trainer.train()
        # Save the final model state  => run.py 에서 저장하자자
        #os.makedirs(args.output_dir, exist_ok=True)
        #self.save_state(os.path.join(args.output_dir, f"{config.get('experiment_name')}.pt"))

    def predict(self, texts):

        self.model.eval()
        toks = self.tokenizer(
            texts, return_tensors='pt', padding=True,
            truncation=True, max_length=self.config['max_length'],
            return_token_type_ids=False
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**toks).logits
        return torch.argmax(logits, dim=1).cpu().tolist()

    def load_state(self, path):
        pretrained = path
        num_labels = self.config["num_labels"]

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=num_labels
        ).to(self.device)

    def save_state(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def evaluate_model(self, texts, labels):
        preds = self.predict(texts)
        return compute_metrics(labels, preds)
