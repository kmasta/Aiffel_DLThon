import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("original_train.csv")

# 매핑 사전 생성
class_mapping = {
    "협박 대화": 0,
    "갈취 대화": 1,
    "직장 내 괴롭힘 대화": 2,
    "기타 괴롭힘 대화": 3,
}

# class 컬럼 값을 매핑에 따라 변환
df["class"] = df["class"].map(class_mapping)

# 변환된 CSV 저장
df.to_csv("numeric_train.csv", index=False)
