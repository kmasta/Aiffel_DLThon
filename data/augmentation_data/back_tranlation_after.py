import pandas as pd


def replace_newlines_to_blank(text):
    """텍스트의 모든 줄바꿈 문자(\n)를 공백으로 치환합니다."""
    if text is None:
        return None
    return text.replace('\n', '')


def check_translation(text:str):
    """augmented_conversation 필드 값을 검사하여 특정 문구가 포함되어 있는지 확인합니다."""
    if text is None:
        return False
    return not text.startswith('죄송합니다만,')


def main():
    df = pd.read_csv("../augmented_data/back_translation_augmented_data.csv")
    print(f"데이터셋 크기: {df.shape}")
    print("데이터프레임 컬럼:", df.columns.tolist())
    augment_target_column = "augmented_conversation"

    # 줄바꿈 제거
    df[augment_target_column] = df[augment_target_column].apply(replace_newlines_to_blank)

    # translation_check 필드 추가 - 특정 문구가 포함되어 있지 않으면 True, 포함되어 있으면 False
    df['translation_check'] = df[augment_target_column].apply(check_translation)

    # 결과 확인 - translation_check가 False인 행의 수
    false_count = len(df[df['translation_check'] == False])
    print(f"translation_check가 False인 행의 수: {false_count}")

    # CSV 파일로 저장
    df.to_csv("../augmented_data/back_translation_augmented_data.csv", index=False)
    print("파일 저장 완료")


if __name__ == "__main__":
    main()