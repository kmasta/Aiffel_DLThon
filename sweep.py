import os
import wandb
import yaml
import argparse
import numpy as np
from datetime import datetime


# wandb 초기화 및 설정 로드
def init_wandb_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="기본 설정 파일 경로")
    parser.add_argument("--sweep_config", default="configs/sweep_config.yaml", help="sweep 설정 파일 경로")
    parser.add_argument("--count", type=int, default=10, help="실행할 sweep 실험 횟수")
    args = parser.parse_args()

    # sweep 설정 로드
    with open(args.sweep_config, "r", encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)

    # 기본 설정 로드
    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    project_name = base_config.get("wandb_project", "text-classification")

    # sweep 초기화
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    return sweep_id, args


def train_with_wandb(config=None):
    # 안전하게 import 하기
    from src.utils import load_config, seed_everything, load_data, encode_labels, train_val_test_split
    from src.models.model import load_model
    from src.metrics import compute_metrics

    with wandb.init(config=config) as run:
        # wandb에서 가져온 config 적용
        wandb_config = wandb.config

        # 기본 설정 로드
        base_config = load_config(args.config)

        # wandb config에서 가져온 설정으로 base_config 업데이트
        base_config["lr"] = wandb_config.learning_rate
        base_config["batch_size"] = wandb_config.batch_size
        base_config["epochs"] = wandb_config.epochs

        if hasattr(wandb_config, "dropout"):
            base_config["hidden_dropout_prob"] = wandb_config.dropout
            base_config["attention_probs_dropout_prob"] = wandb_config.dropout

        if hasattr(wandb_config, "optimizer"):
            base_config["optimizer"] = wandb_config.optimizer

        # 실험 이름 설정 (wandb run ID 포함)
        run_id = run.id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_config["experiment_name"] = f"{base_config.get('experiment_name', 'exp')}_{run_id}_{timestamp}"

        # 시드 고정
        seed_everything(base_config.get("seed", 42))

        # 모델 로드
        model = load_model(base_config)

        # 데이터 로드 및 인코딩
        texts, labels = load_data(base_config["train_data_path"])
        labels_enc = encode_labels(labels, config=base_config)

        # 데이터 분할
        if base_config.get("save_submission", True):
            train_texts, train_labels, val_texts, val_labels, _, _ = train_val_test_split(
                texts, labels_enc,
                val_ratio=base_config.get("val_ratio", 0.2),
                test_ratio=0.0,
                seed=base_config.get("seed", 42),
            )
        else:
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
                texts, labels_enc,
                val_ratio=base_config.get("val_ratio", 0.2),
                test_ratio=base_config.get("test_ratio", 0.1),
                seed=base_config.get("seed", 42),
            )

        # 모델 훈련
        if base_config['framework'].lower() == 'pytorch':
            best_epoch = model.train_model(
                train_texts, train_labels,
                val_texts, val_labels,
                base_config
            )
            wandb.log({"best_epoch": best_epoch})
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

        # 테스트 데이터가 있는 경우 테스트 성능도 평가
        if not base_config.get("save_submission", True):
            test_preds = model.predict(test_texts)
            test_metrics = compute_metrics(test_labels, test_preds)

            wandb.log({
                "test/accuracy": test_metrics["accuracy"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/f1": test_metrics["f1"]
            })


if __name__ == "__main__":
    # wandb sweep 초기화
    sweep_id, args = init_wandb_sweep()

    # sweep 실행
    wandb.agent(sweep_id, train_with_wandb, count=args.count)