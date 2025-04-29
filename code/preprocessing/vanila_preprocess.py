import re

# 전처리 함수
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
    sentence = re.sub(r'\s+', ' ', sentence)

    # 특수문자 정리
    sentence = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9?.!,]+", " ", sentence)

    return sentence.strip()
