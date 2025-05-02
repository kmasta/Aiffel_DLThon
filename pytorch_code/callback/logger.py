import os
import datetime
import pandas as pd

def log_message(message: str, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")

    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
    print(f"{timestamp} {message}")

def log_target_distribution(df: pd.DataFrame, log_dir: str):
    """
    주어진 DataFrame에서 'target' 컬럼의 클래스 분포를 로그 파일과 콘솔에 출력합니다.
    """
    if "target" not in df.columns:
        log_message("❌ 'target' 컬럼이 데이터프레임에 없습니다.", log_dir)
        return

    counts = df["target"].value_counts().sort_index()
    log_message("📊 Target 클래스 분포:", log_dir)
    for label, count in counts.items():
        log_message(f"Label {label}: {count}", log_dir)

        
from transformers import TrainerCallback

def log_training_metrics(epoch: int, train_loss: float, val_loss: float, val_acc: float, val_f1: float, log_dir: str):
    """
    한 에폭의 학습 지표를 로그 파일과 콘솔에 기록합니다.
    """
    message = (
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1(Macro): {val_f1:.4f}"
    )
    log_message(message, log_dir)

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.epoch is None:
            return

        # 로그에서 지표 가져오기
        epoch = int(state.epoch)
        train_loss = logs.get("loss", None)
        val_loss = logs.get("eval_loss", None)
        val_acc = logs.get("eval_accuracy", None)
        val_f1 = logs.get("eval_f1_macro", None)

        # validation이 아닌 경우는 생략
        if val_loss is not None:
            log_training_metrics(
                epoch,
                train_loss if train_loss is not None else 0,
                val_loss,
                val_acc if val_acc is not None else 0,
                val_f1 if val_f1 is not None else 0,
                args.logging_dir,
            )
