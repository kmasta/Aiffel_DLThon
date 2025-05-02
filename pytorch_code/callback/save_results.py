import pandas as pd
import os

def save_submission(preds, submission_file_path, output_path):
    """
    예측 결과를 sample_submission.csv 형식으로 저장
    - file_name → idx
    - class → target (원본)
    - class 열에는 예측값(preds) 삽입
    """
    submission = pd.read_csv(submission_file_path)

    # 컬럼 이름 변경
    submission = submission.rename(columns={"file_name": "idx", "class": "target"})

    # target 열에 예측값 삽입
    submission["target"] = preds

    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
