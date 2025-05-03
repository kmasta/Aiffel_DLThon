import os
os.environ["TRANSFORMERS_NO_INTEGRATION"] = "1"
import argparse
import sys
import logging
import shutil
import csv
import builtins
from datetime import datetime


import torch
from src.utils import (
    seed_everything, load_config, load_data,
    train_val_test_split, encode_labels,
    save_submission, load_texts
)
from src.metrics import compute_metrics, category_metrics
from src.models.model import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    # 설정 로드 및 시드 고정
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    # 모델 생성
    model      = load_model(config)
    # 로그 파일 경로 준비
    exp_name = config.get("experiment_name", config["model_name"])
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/{exp_name}_{ts}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 파일 핸들러: Python 로그 전용
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # 콘솔 핸들러: stdout/stderr 로만
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    # 포맷터
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    # 핸들러 등록 (중복 방지)
    logger.handlers = [fh, sh]

    # print() 래핑
    _orig_print = builtins.print
    def _print_via_logging(*args, **kwargs):
        _orig_print(*args, **kwargs)
        logging.info(" ".join(str(a) for a in args))
    builtins.print = _print_via_logging

    # stderr.write 래핑 (traceback 포함)
    _orig_stderr_write = sys.stderr.write
    def _stderr_via_logging(msg):
        _orig_stderr_write(msg)
        for line in msg.rstrip().splitlines():
            logging.error(line)
    sys.stderr.write = _stderr_via_logging

    # uncaught 예외 로깅
    def _exception_logger(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", 
                    exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = _exception_logger

    logging.info(f"Experiment start: {exp_name}")
    logging.info(f"Config: {args.config}")

    # 데이터 로드 및 인코딩
    texts, labels = load_data(config["train_data_path"])
    labels_enc    = encode_labels(labels, config=config)

    # 데이터 분할
    if config.get("save_submission", True):
        train_texts, train_labels, val_texts, val_labels, _, _ = train_val_test_split(
            texts, labels_enc,
            val_ratio  = config.get("val_ratio", 0.2),
            test_ratio = 0.0,
            seed       = config.get("seed", 42),
        )
        predict_texts = load_texts(config["predict_data_path"])
    else:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
            texts, labels_enc,
            val_ratio  = config.get("val_ratio", 0.2),
            test_ratio = config.get("test_ratio", 0.1),
            seed       = config.get("seed", 42),
        )



    # 모델 훈련
    if not config.get("use_only_eval", False):
        # 모델 훈련
        if config['framework'].lower() == 'pytorch':
            # Hugging Face Trainer 기반 학습
            model.train_model(
                train_texts, train_labels,
                val_texts, val_labels,
                config
            )
        else:
            # Keras 모델 학습 (콜백 로깅 유지)
            model.train_model(
                train_texts, train_labels,
                val_texts, val_labels,
                config, log_callback=True
            )
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_base = f"checkpoints/{exp_name}_{ts}"
        model.save_state(ckpt_base)
        shutil.copy(args.config, ckpt_base + "_config.yaml")
        logging.info(f"Model checkpoint saved to {ckpt_base}")

    # 결과 기록용 CSV
    os.makedirs("experiments", exist_ok=True)
    res_path = config.get("result_file_path", "experiments/results.csv")


    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    write_header = not os.path.isfile(res_path)

    with open(res_path, "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)

        if write_header:
            header = ["exp_name", "ts", "split", "n_samples", "loss_fn", "accuracy", "precision", "recall", "f1"]
            for i in range(config["num_labels"]):
                header += [f"class_{i}_accuracy", f"class_{i}_f1", f"class_{i}_pred_count"]
            writer.writerow(header)

        def evaluate_and_write(split, xs, ys=None):
            preds = model.predict(xs)
            base = [exp_name, ts]

            if ys is not None:
                m = compute_metrics(ys, preds)
                cm = category_metrics(ys, preds)
                pred_counts = [preds.count(i) for i in range(config["num_labels"])]

                logging.info(f"[{split}] accuracy={m['accuracy']:.4f}, precision={m['precision']:.4f}, "
                            f"recall={m['recall']:.4f}, f1={m['f1']:.4f}")
                for i in range(config["num_labels"]):
                    logging.info(f"    class_{i}: accuracy={cm[f'class_{i}_accuracy']:.4f}, "
                                f"f1={cm[f'class_{i}_f1']:.4f}, count={pred_counts[i]}")

                row = base + [split, len(ys), config.get("loss_fn", "")]
                row += [m["accuracy"], m["precision"], m["recall"], m["f1"]]
                for i in range(config["num_labels"]):
                    row += [cm[f"class_{i}_accuracy"], cm[f"class_{i}_f1"], pred_counts[i]]
            else:
                # submission
                pred_counts = [preds.count(i) for i in range(config["num_labels"])]
                logging.info(f"[{split}] num_samples={len(xs)}")
                for i in range(config["num_labels"]):
                    logging.info(f"    class_{i}_pred_count={pred_counts[i]}")

                row = base + [split, len(xs), config.get("loss_fn", "")]
                # 빈 지표 위치(accuracy, precision, recall, f1)
                row += [""] * 4
                # 클래스별 칼럼 수는 3개씩(accuracy, f1, pred_count), 여기선 pred_count만 채우고 나머지는 빈칸
                for i in range(config["num_labels"]):
                    row += ["", ""] + [pred_counts[i]]

            writer.writerow(row)
            return preds


        if config.get("save_submission", True):
            if not config.get("use_only_eval", False):
                evaluate_and_write("val", val_texts, val_labels)
            preds = evaluate_and_write("submission", predict_texts, None)
            sub_dir = f"submissions/{exp_name}_{ts}"
            os.makedirs(sub_dir, exist_ok=True)
            shutil.copy(args.config, sub_dir + f"{exp_name}_{ts}_config.yaml")
            sub_file = f"{sub_dir}/submission_{exp_name}_{ts}.csv"
            save_submission(preds, sub_file)
            logging.info(f"Submission file saved to {sub_file}")
        else:
            if not config.get("use_only_eval", False):
                evaluate_and_write("val", val_texts, val_labels)
            preds = evaluate_and_write("test", test_texts, test_labels)
            shutil.copy(args.config, f"experiments/test_{exp_name}_{ts}_config.yaml")
            test_result_file = f"experiments/test_{exp_name}_{ts}_result.csv"
            save_test_result(test_texts, preds, test_labels, test_result_file)
            logging.info(f"Non-submission test preds saved to {test_result_file}")

    logging.info(f"Results recorded to {res_path}")
    print(f"Results saved: {res_path}")
