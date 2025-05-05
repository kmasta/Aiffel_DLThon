import argparse
import pandas as pd
from eunjeon import Mecab
import os

# 불용어 리스트 (중복 제거)
stopwords = set([
    '이', '그', '저', '것', '수', '들', '등', '때', '문제', '뿐', '안', '이랑', '랑', 
    '도', '곳', '걸', '에서', '하지만', '그렇지만', '그러나', '그리고', '따라서', 
    '그러므로', '그러나', '그런데', '때문에', '왜냐하면', '무엇', '어디', '어떤', 
    '어느', '어떻게', '누가', '누구', '어떤', '한', '하다', '있다', '되다', '이다', 
    '로', '로서', '로써', '과', '와', '이다', '입니다', '한다', '할', '위해', 
    '또한', '및', '이외', '더불어', '그리고', '따라', '따라서', '뿐만아니라', '그럼', 
    '하지만', '있어서', '그래서', '그렇다면', '이에', '때문에', '무엇', '어디', 
    '어떻게', '왜', '어느', '하는', '하게', '해서', '이러한', '이렇게', '그러한', 
    '그렇게', '저러한', '저렇게', '하기', '한것', '한것이', '일때', '있는', '있는것', 
    '있는지', '여기', '저기', '거기', '뭐', '왜', '어디', '어느', '어떻게', '무엇을', 
    '어디서', '어디에', '무엇인가', '무엇이', '어떤', '누가', '누구', '무엇', 
    '어디', '어떤', '한', '하다', '있다', '되다', '이다', '로', '로서', '로써', 
    '과', '와', '이', '그', '저', '것', '수', '들', '등', '때', '문제', '뿐', 
    '안', '이랑', '랑', '도', '곳', '걸', '에서', '하지만', '그렇지만', '그러나', 
    '그리고', '따라서', '그러므로', '그러나', '그런데', '때문에', '왜냐하면'
])

mecab = Mecab()

def remove_stopwords(texts, stopwords):
    results = []
    for text in texts:
        if not isinstance(text, str):
            results.append("")
            continue
        tokens = mecab.morphs(text)
        filtered_tokens = [token for token in tokens if token not in stopwords]
        results.append(' '.join(filtered_tokens))
    return results

def main(args):
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 디렉토리 생성
    
    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"입력 파일에 '{args.text_col}' 열이 존재하지 않습니다.")
    
    print("✅ 불용어 제거 중...")
    df[args.text_col] = remove_stopwords(df[args.text_col], stopwords)
    df.to_csv(args.output, index=False)
    print(f"✅ 저장 완료: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="불용어 제거 전처리 (Okt + 불용어 사전 기반)")
    parser.add_argument('--input', required=True, help='입력 CSV 경로')
    parser.add_argument('--output', required=True, help='출력 CSV 경로')
    parser.add_argument('--text_col', default='clean_text', help='불용어 제거할 열 이름 (기본값: clean_text)')
    args = parser.parse_args()
    main(args)
