import os
import datetime

def log_message(message: str, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")

    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
    print(f"{timestamp} {message}")
