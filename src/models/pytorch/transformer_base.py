import os
import re
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from src.models.base import BaseModel
from src.metrics import compute_metrics
from transformers.trainer_callback import TrainerCallback
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import LBFGS
from transformers import TrainerCallback

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # log-space로 학습하는 게 안정적이므로, 실제 T = exp(log_temp)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        # temperature = exp(log_temp)
        temp = torch.exp(self.log_temp)
        return logits / temp

def fit_temperature(logits, labels, device="cpu"):
    """
    logits: torch.Tensor, shape (N, C)
    labels: torch.Tensor, shape (N,)
    returns: optimal temperature scalar (float)
    """
    # move to device
    logits = logits.to(device)
    labels = labels.to(device)

    scaler = TemperatureScaler().to(device)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = LBFGS([scaler.log_temp], lr=0.1, max_iter=50)

    def _eval():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(_eval)
    return float(torch.exp(scaler.log_temp).detach().cpu())

class TemperatureCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.trainer = None

    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        # 1) eval dataloader 얻기
        eval_dataloader = trainer.get_eval_dataloader()

        # 2) 전체 validation logits, labels 수집
        all_logits = []
        all_labels = []
        trainer.model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(trainer.model.device) for k, v in batch.items()}
                outputs = trainer.model(**batch)
                all_logits.append(outputs.logits)
                all_labels.append(batch["labels"])
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # 3) 온도 학습 및 저장
        T_opt = fit_temperature(logits, labels, device=trainer.model.device)
        print(f"[Temperature Scaling] optimal T = {T_opt:.3f}")
        #trainer.model.config.temperature = T_opt


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False  # ← 여기서 token_type_ids 생성 멈춤
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
        if val_texts:
            eval_dataset = TextDataset(val_texts, val_labels, self.tokenizer, config['max_length'])
        else:
            eval_dataset = None

        raw_name = config.get('experiment_name', 'exp_noname')
        exp_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_name)

        # Define training arguments
        args = TrainingArguments(
            output_dir=os.path.join('checkpoints', exp_name),
            num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            # eval_strategy='epoch',
            logging_dir=os.path.join('logs', exp_name),
            # logging_strategy='steps',
            # logging_steps=config.get('logging_steps', 100),
            optim      = config.get("optimizer", "adamw_torch"),           # Optimizer 종류 adafactor
            optim_args = config.get("optim_args", None),               # Optimizer 초기화 인자
            learning_rate=float(config['lr']),  # 초기 학습률
            lr_scheduler_type=config.get("lr_scheduler_type", "linear"),    # "linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"
            warmup_steps=config.get("warmup_steps", 0),
            warmup_ratio=config.get("warmup_ratio", 0.0),
            weight_decay=config.get("weight_decay", 0),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            label_smoothing_factor=config.get("label_smoothing", 0.0),
            eval_strategy="epoch", # Transformer Version이 낮을 경우에는 evaluation_strategy로 변경
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_strategy='epoch',
            save_total_limit=3,
            seed=config.get('seed', 42),
            report_to=[],
            remove_unused_columns=False
        )
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)

        temp_cb = TemperatureCallback()


        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset is not None else train_dataset,
            compute_metrics=lambda p: compute_metrics(p.label_ids, p.predictions.argmax(-1)),
            callbacks=[temp_cb, EarlyStoppingCallback(early_stopping_patience=config.get("patience", 3))]
        )
        temp_cb.trainer = trainer


        # Train the model
        trainer_output  = trainer.train()
        # Save the final model state  => run.py 에서 저장하자자

        # 1) best_model_checkpoint 경로에서 step 정보 파싱
        best_ckpt = trainer.state.best_model_checkpoint  # e.g. "checkpoints/exp/checkpoint-500"
        if best_ckpt:
            m = re.search(r"checkpoint-(\d+)", best_ckpt)
            if m:
                best_step = int(m.group(1))
                # 2) 스텝→epoch 계산 (dataset 크기, batch size 기준)
                steps_per_epoch = len(train_dataset) // args.per_device_train_batch_size
                best_epoch = best_step // steps_per_epoch
            else:
                best_epoch = int(trainer.state.epoch)
        else:
            best_epoch = int(trainer.state.epoch)

        return best_epoch

    def predict(self, texts):
        self.model.eval()
        toks = self.tokenizer(
            texts, return_tensors='pt', padding=True,
            truncation=True, max_length=self.config['max_length'],
            return_token_type_ids=False
        )

        # 토큰을 장치로 옮기기 전에 token_type_ids 제거
        if 'token_type_ids' in toks:
            toks.pop('token_type_ids')

        try:
            # MPS 디바이스에 문제가 있는 경우 CPU로 폴백
            if self.device.type == 'mps':
                try:
                    # 장치로 이동 시도
                    toks = {k: v.to(self.device) for k, v in toks.items()}
                    with torch.no_grad():
                        logits = self.model(**toks).logits
                except RuntimeError:
                    # MPS 오류 발생 시 CPU로 대체
                    print("MPS 디바이스 오류 발생, CPU로 전환합니다.")
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
                    toks = {k: v.to(self.device) for k, v in toks.items()}
                    with torch.no_grad():
                        logits = self.model(**toks).logits
            else:
                # 기존 장치 사용(CUDA 또는 CPU)
                toks = {k: v.to(self.device) for k, v in toks.items()}
                with torch.no_grad():
                    logits = self.model(**toks).logits
        except Exception as e:
            # 그 외 오류 시 CPU로 안전하게 폴백
            print(f"예측 중 오류 발생: {e}, CPU로 전환합니다.")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                logits = self.model(**toks).logits

        T     = getattr(self.model.config, "temperature", 1.0)
        probs = torch.softmax(logits / T, dim=-1)
        preds = torch.argmax(probs, dim=1)

        return preds.cpu().tolist()

    def predict_proba(self, texts):
        """
        앙상블 소프트보팅용: 입력 texts에 대한
        클래스별 확률 분포 (N, num_labels) 배열을 반환합니다.
        """
        self.model.eval()
        # 토크나이저 인퍼런스
        toks = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['max_length'],
            return_token_type_ids=False
        )
        # 토큰을 장치로 옮기기 전에 token_type_ids 제거
        if 'token_type_ids' in toks:
            toks.pop('token_type_ids')

        try:
            # MPS 디바이스에 문제가 있는 경우 CPU로 폴백
            if self.device.type == 'mps':
                try:
                    # 장치로 이동 시도
                    toks = {k: v.to(self.device) for k, v in toks.items()}
                    with torch.no_grad():
                        logits = self.model(**toks).logits
                except RuntimeError:
                    # MPS 오류 발생 시 CPU로 대체
                    print("MPS 디바이스 오류 발생, CPU로 전환합니다.")
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
                    toks = {k: v.to(self.device) for k, v in toks.items()}
                    with torch.no_grad():
                        logits = self.model(**toks).logits
            else:
                # 기존 장치 사용(CUDA 또는 CPU)
                toks = {k: v.to(self.device) for k, v in toks.items()}
                with torch.no_grad():
                    logits = self.model(**toks).logits
        except Exception as e:
            # 그 외 오류 시 CPU로 안전하게 폴백
            print(f"예측 중 오류 발생: {e}, CPU로 전환합니다.")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                logits = self.model(**toks).logits

        # softmax로 확률 분포로 변환
        T     = getattr(self.model.config, "temperature", 1.0)
        probs = torch.softmax(logits / T, dim=-1)
        return probs.cpu().numpy()

    def load_state(self, path):
        pretrained = path
        num_labels = self.config["num_labels"]


        # 로컬 디렉터리인지 감지
        is_local = Path(pretrained).is_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            local_files_only=is_local
        )

        config = AutoConfig.from_pretrained(
            pretrained,
            num_labels=num_labels,
            local_files_only=is_local
        )

        config.hidden_dropout_prob = self.config.get("hidden_dropout_prob", 0.1)
        config.attention_probs_dropout_prob = self.config.get("attention_probs_dropout_prob", 0.1)

        # 디바이스 선택 로직 개선
        if torch.backends.mps.is_available():
            try:
                # MPS 가용성 테스트
                test_tensor = torch.zeros(1).to("mps")
                self.device = torch.device("mps")
                print("MPS 디바이스를 사용합니다.")
            except Exception as e:
                print(f"MPS 초기화 실패: {e}, CPU를 사용합니다.")
                self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA 디바이스를 사용합니다.")
        else:
            self.device = torch.device("cpu")
            print("CPU 디바이스를 사용합니다.")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained,
            config=config,
            local_files_only=is_local
        ).to(self.device)

    def save_state(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def evaluate_model(self, texts, labels):
        preds = self.predict(texts)
        return compute_metrics(labels, preds)