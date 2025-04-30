import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# 대화 전처리 함수
def preprocess_sentence(sentence):
    sentence = sentence.strip().lower()

    # 구두점 앞뒤로 띄어쓰기
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)

    # 숫자와 문자 사이 띄어쓰기
    sentence = re.sub(r"(\d)([가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z])", r"\1 \2", sentence)
    sentence = re.sub(r"([가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z])(\d)", r"\1 \2", sentence)

    # 영어-한글 또는 한글-영어 사이 띄어쓰기
    sentence = re.sub(r"([a-zA-Z])([가-힣ㄱ-ㅎㅏ-ㅣ])", r"\1 \2", sentence)
    sentence = re.sub(r"([가-힣ㄱ-ㅎㅏ-ㅣ])([a-zA-Z])", r"\1 \2", sentence)

    # 자모 앞에 공백 삽입
    sentence = re.sub(r"(?<=[가-힣a-zA-Z0-9])(?=[ㄱ-ㅎㅏ-ㅣ]+)", " ", sentence)

    # 여러 공백 정리
    sentence = re.sub(r"\s+", " ", sentence)

    # 특수문자 정리
    sentence = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9?.!,]+", " ", sentence)

    return sentence.strip()

# 고정된 라벨 매핑
label_to_id = {
    "협박 대화": 0,
    "갈취 대화": 1,
    "직장 내 괴롭힘 대화": 2,
    "기타 괴롭힘 대화": 3,
    "일반 대화": 4
}
id_to_label = {v: k for k, v in label_to_id.items()}

def encode_labels(labels, method="int"):
    """
    labels: 한글 라벨 리스트
    method: "int" 또는 "onehot"
    """
    int_labels = [label_to_id[label] for label in labels]

    if method == "int":
        return int_labels
    elif method == "onehot":
        return to_categorical(int_labels, num_classes=len(label_to_id))
    else:
        raise ValueError("지원하지 않는 method입니다. 'int' 또는 'onehot'만 사용하세요.")

def decode_labels(int_labels):
    """정수 라벨 → 한글 라벨"""
    return [id_to_label[i] for i in int_labels]
