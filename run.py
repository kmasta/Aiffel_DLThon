import os
os.environ["TRANSFORMERS_NO_INTEGRATION"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import sys
import logging
import shutil
import csv
import builtins
from datetime import datetime


import torch
from src.utils import *
from src.metrics import *
from src.models.model import *
from src.ensemble import *

import yaml       # sub-config 로드용
import copy       # deep copy
import numpy as np
from joblib import dump, load   # stacking meta model 저장/로드
from sklearn.model_selection import train_test_split, StratifiedKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()


    # 설정 로드 및 시드 고정
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))


    is_ensemble = bool(config.get("ensemble"))
    ens_cfg     = config.get("ensemble", {})
    # single-config와 sub-config 모두에서 use_only_eval 읽기
    #use_only_eval_top = config.get("use_only_eval", False)


    if is_ensemble:
        sub_models = []     # 리스트에 튜플(model, sub_cfg, weight) 저장
        for sub in ens_cfg["models"]:
            # 1) 기존 single-config 로드
            with open(sub["config_path"], "r", encoding="utf-8") as f:
                sub_cfg = yaml.safe_load(f)

            # 2) 무한 재귀 방지
            sub_cfg.pop("ensemble", None)

            # 3) 모델 객체 생성 (pretrained_model_name 로드 포함)
            m = load_model(sub_cfg)

            # 4) 학습 스킵 플래그: sub_cfg 에 있으면 그걸 사용, 없으면 top-level 사용
            skip_train = sub_cfg.get("use_only_eval", False)

            sub_models.append((m, sub_cfg, sub.get("weight", 1.0), skip_train))
    else:
        model = load_model(config)
        skip_train = config.get("use_only_eval", False)

    # 로그 파일 경로 준비
    raw_name = config.get("experiment_name", "exp_noname")
    exp_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_name)
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
    #_orig_stderr_write = sys.stderr.write
    #def _stderr_via_logging(msg):
    #    _orig_stderr_write(msg)
    #    for line in msg.rstrip().splitlines():
    #        logging.error(line)
    #sys.stderr.write = _stderr_via_logging

    ## uncaught 예외 로깅
    #def _exception_logger(exc_type, exc_value, exc_traceback):
    #    logger.error("Uncaught exception",
    #                 exc_info=(exc_type, exc_value, exc_traceback))
    #sys.excepthook = _exception_logger

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
    # 기존 random seed 분할이 유지 되도록 기존 코드 유지 후 fold 분할
    val_cfg = config.get("validation", {})

    pool_texts = train_texts
    pool_labels = train_labels
    if val_cfg.get("strategy") == "kfold":
        # train+val 풀 결합
        pool_texts  = train_texts + val_texts
        pool_labels = np.concatenate([train_labels, val_labels])
        skf = StratifiedKFold(
            n_splits      = val_cfg.get("folds", 5),
            shuffle       = True,
            random_state  = config.get("seed")
        )
        folds_data = [
            (
                [pool_texts[i] for i in tr],
                [pool_labels[i] for i in tr],
                [pool_texts[i] for i in va],
                [pool_labels[i] for i in va],
            )
            for tr, va in skf.split(pool_texts, pool_labels)
        ]
    else:
        # 한 번만 split
        folds_data = [(train_texts, train_labels, val_texts, val_labels)]

    # 모델 훈련
    os.makedirs("checkpoints", exist_ok=True)
    mode = (
        "stacking" if is_ensemble and ens_cfg.get("strategy")=="stacking"
        else "ensemble" if is_ensemble
        else "single"
    )
    best_epochs = []  # 각 fold 최적 에폭 기록
    oof_f1s = []
    # 에러 샘플 기록용 파일 경로
    err_path = f"experiments/{exp_name}_{ts}_val_errors.csv"
    # 헤더 쓰기 (처음 한 번만)
    if not os.path.exists(err_path):
        with open(err_path, 'w', newline='', encoding='utf-8') as ef:
            w = csv.writer(ef)
            w.writerow(["exp_name","ts","mode","fold","text","true_label","pred_label"])


    for fold_idx, (tr_x, tr_y, va_x, va_y) in enumerate(folds_data, start=1):
        logging.info(f"[{mode}][Fold {fold_idx}/{len(folds_data)}] Training…")
        if is_ensemble:
            for m, sub_cfg, weight, skip_train in sub_models:
                if skip_train:
                    logger.info(f"[{sub_cfg['experiment_name']}] use_only_eval=True → 훈련 SKIP")
                    continue
                else:
                    if config['framework'].lower() == 'pytorch':
                        m.train_model(tr_x, tr_y, va_x, va_y, sub_cfg)
                    else:
                        m.train_model(
                            train_texts, train_labels,
                            val_texts, val_labels,
                            sub_cfg, log_callback=True
                        )  #  keras - 지원안될수 있음

                    # ── submodel checkpoint 저장
                    ckpt_base = f"checkpoints/{sub_cfg['experiment_name']}_{ts}"
                    if val_cfg.get("strategy") != "kfold":
                        m.save_state(ckpt_base)
                        shutil.copy(sub['config_path'], ckpt_base + "_config.yaml")
                        logging.info(f"[{sub_cfg['experiment_name']}] checkpoint saved to {ckpt_base}")
            preds = get_preds(
                [m for m, *_ in sub_models],
                [w for *_, w, _ in sub_models],
                ens_cfg.get("strategy","weighted_soft"),
                ens_cfg["meta_path"],
                va_x
            )
        else:
            if not config.get("use_only_eval", False):
                # 모델 훈련
                if config['framework'].lower() == 'pytorch':
                    # Hugging Face Trainer 기반 학습
                    be = model.train_model(
                        train_texts, train_labels,
                        val_texts, val_labels,
                        config
                    )
                    best_epochs.append(be)
                else:
                    # Keras 모델 학습 (콜백 로깅 유지)
                    model.train_model(
                        train_texts, train_labels,
                        val_texts, val_labels,
                        config, log_callback=True
                    )
                # ── single-model checkpoint 저장
                ckpt_base = f"checkpoints/{exp_name}_{ts}"
                model.save_state(ckpt_base)               # ### ORIGINAL-CALL
                shutil.copy(args.config, ckpt_base + "_config.yaml")
                logging.info(f"Model checkpoint saved to {ckpt_base}")
            else:
                logger.info(f"[{config['experiment_name']}] use_only_eval=True → 훈련 SKIP")

            preds = model.predict(va_x)
            # Metrics & 기록
            m_val = compute_metrics(va_y, preds)
            oof_f1s.append(m_val["f1"])
            # TODO evaluate_and_write(mode, fold_idx, "val", va_x, va_y)

            # 에러 샘플 저장 (각 fold의 validation에서 오분류된 샘플)
        with open(err_path, 'a', newline='', encoding='utf-8') as ef:
            w = csv.writer(ef)
            for text, true, pred in zip(va_x, va_y, preds):
                if true != pred:
                    w.writerow([exp_name, ts, mode, fold_idx, text, true, pred])


    # ---------------- 스태킹 메타 모델 학습 ----------------
    if is_ensemble and ens_cfg.get("strategy") == "stacking" and not config.get("use_only_eval", False):
            # OOF 전체 합쳐서 meta 학습
        all_va_x = sum([fd[2] for fd in folds_data], [])
        all_va_y = sum([fd[3] for fd in folds_data], [])
        meta_model = train_meta(
            [m for m, *_ in sub_models],
            all_va_x,
            all_va_y,
            "stacking",
            ens_cfg["meta_path"]
            )
    else:
        if not config.get("use_only_eval", False):
            if mode == "single":
                logging.info("Full-training on pool → evaluating holdout test")
                full_model = load_model(config)
                if not skip_train:
                    config_full = config.copy()
                    config_full['epochs'] = int(np.mean(best_epochs))  # CV 단계 평균 best epoch
                    full_model.train_model(pool_texts, pool_labels, [], [], config_full)
                test_preds = full_model.predict(test_texts)

                # fold=0로 holdout 결과 기록
                # TODO evaluate_and_write(mode, 0, "test", test_texts, test_labels)

                # 오분류 샘플 저장 (풀 학습 후 holdout 테스트)
                err_test_path = f"experiments/{exp_name}_{ts}_test_errors.csv"
                if not os.path.exists(err_test_path):
                    with open(err_test_path, 'w', newline='', encoding='utf-8') as ef:
                        w = csv.writer(ef)
                        w.writerow(["exp_name","ts","mode","fold","text","true_label","pred_label"])
                with open(err_test_path, 'a', newline='', encoding='utf-8') as ef:
                    w = csv.writer(ef)
                    for text, true, pred in zip(test_texts, test_labels, test_preds):
                        if true != pred:
                            w.writerow([exp_name, ts, mode, 0, text, true, pred])
                logging.info(f"OOF mean F1: {np.mean(oof_f1s):.4f}, Holdout F1: {compute_metrics(test_labels, test_preds)['f1']:.4f}, Gap: {abs(np.mean(oof_f1s)-compute_metrics(test_labels, test_preds)['f1']):.4f}")

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

        def evaluate_and_write(mode, fold_idx, split, xs, ys=None):
            if is_ensemble:
                preds = get_preds(
                    [m for m, *_ in sub_models],
                    [w for *_, w, _ in sub_models],
                    ens_cfg.get("strategy","weighted_soft"),
                    ens_cfg["meta_path"],
                    xs
                )
            else:
                if val_cfg.get("strategy") == "kfold":
                    preds = full_model.predict(xs)
                else:
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
                evaluate_and_write(mode, fold_idx,"val", val_texts, val_labels)
            preds = evaluate_and_write(mode, fold_idx,"submission", predict_texts, None)
            sub_dir = f"submissions/{exp_name}_{ts}"
            os.makedirs(sub_dir, exist_ok=True)
            shutil.copy(args.config, sub_dir + f"{exp_name}_{ts}_config.yaml")
            sub_file = f"{sub_dir}/submission_{exp_name}_{ts}.csv"
            save_submission(preds, sub_file)
            logging.info(f"Submission file saved to {sub_file}")
        else:
            if not config.get("use_only_eval", False):
                evaluate_and_write(mode, fold_idx,"val", val_texts, val_labels)
            preds = evaluate_and_write(mode, fold_idx,"test", test_texts, test_labels)
            shutil.copy(args.config, f"experiments/test_{exp_name}_{ts}_config.yaml")
            test_result_file = f"experiments/test_{exp_name}_{ts}_result.csv"
            save_test_result(test_texts, preds, test_labels, test_result_file)
            logging.info(f"Non-submission test preds saved to {test_result_file}")

    logging.info(f"Results recorded to {res_path}")
    print(f"Results saved: {res_path}")
    