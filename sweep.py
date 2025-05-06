import os
import wandb
import yaml
import argparse
import numpy as np
from src.utils import load_config, seed_everything, load_data, encode_labels
from src.models.model import load_model
from src.metrics import compute_metrics
from sklearn.model_selection import train_test_split


def train_with_wandb(config=None):
    with wandb.init(config=config):
        # wandb에서 가져온 config 적용
        config = wandb.config

        # 기본 설정 로드
        base_config = load_config(args.config)

        # wandb config에서 가져온 설정으로 base_config 업데이트
        base_config["lr"] = config.learning_rate
        base_config["batch_size"] = config.batch_size
        base_config["epochs"] = config.epochs

        if hasattr(config, "dropout"):
            base_config["hidden_dropout_prob"] = config.dropout
            base_config["attention_probs_dropout_prob"] = config.dropout

        if hasattr(config, "optimizer"):
            base_config["optimizer"] = config.optimizer

        # 시드 고정
        seed_everything(base_config.get("seed", 42))

        # 데이터 로드 및 인코딩
        texts, labels = load_data(base_config["train_data_path"])
        labels_enc = encode_labels(labels, config=base_config)

        # 데이터 분할
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels_enc,
            test_size=base_config.get("val_ratio", 0.2),
            random_state=base_config.get("seed", 42),
        )

        # 모델 로드
        model = load_model(base_config)

        # 모델 훈련
        if base_config['framework'].lower() == 'pytorch':
            best_epoch = model.train_model(
                train_texts, train_labels,
                val_texts, val_labels,
                base_config
            )
        else:
            model.train_model(
                train_texts, train_labels,
                val_texts, val_labels,
                base_config, log_callback=True
            )

        # 검증 데이터로 성능 평가
        preds = model.predict(val_texts)
        metrics = compute_metrics(val_labels, preds)

        # wandb에 결과 로깅
        wandb.log({
            "val/accuracy": metrics["accuracy"],
            "val/precision": metrics["precision"],
            "val/recall": metrics["recall"],
            "val/f1": metrics["f1"]
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="기본 설정 파일 경로")
    parser.add_argument("--sweep_config", default="configs/sweep_config.yaml", help="sweep 설정 파일 경로")
    parser.add_argument("--count", type=int, default=10, help="실행할 sweep 실험 횟수")
    args = parser.parse_args()

    # sweep 설정 로드
    with open(args.sweep_config, "r", encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)

    # wandb 프로젝트 설정
    base_config = load_config(args.config)
    project_name = base_config.get("wandb_project", "text-classification-final")

    # sweep 초기화
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    # sweep 실행
    wandb.agent(sweep_id, train_with_wandb, count=args.count)