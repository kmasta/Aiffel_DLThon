import pandas as pd
from hanspell import spell_checker
from hanspell.response import Checked
from tqdm import tqdm


def hanspell_augmentation(text: str) -> Checked:
    result = spell_checker.check(text)
    return result


def main():
    df = pd.read_csv("../original_data/numeric_train.csv")
    print(f"데이터셋 크기: {df.shape}")
    print("데이터프레임 컬럼:", df.columns.tolist())
    augment_target_column = "conversation"
    augment_result_column = "augmented_conversation"
    spell_checked_column = "spell_checked"

    df_back_translation = df.copy()
    texts = df[augment_target_column].tolist()

    # 전체 텍스트 수에 대한 진행 상황 표시를 위한 tqdm 객체 생성
    with tqdm(total=len(texts), desc="텍스트 처리 중") as pbar:
        augmented_texts = []
        spell_checked = []

        # 배치 단위로 처리
        for i in tqdm(texts):

            batch_result = hanspell_augmentation(i.replace("\n", " "))
            # extend() 대신 append() 사용
            if batch_result.result:
                augmented_texts.append(batch_result.checked)
                spell_checked.append(batch_result.result)
            else:
                augmented_texts.append(i)
                spell_checked.append(batch_result.result)
            pbar.update(1)  # 진행 상황 업데이트

        # 결과를 데이터프레임에 추가
        df_back_translation[augment_result_column] = augmented_texts
        df_back_translation[spell_checked_column] = spell_checked

        # 결과 확인
        print(f"원본 텍스트 예시: {df[augment_target_column][0]}")
        print(f"증강된 텍스트 예시: {df_back_translation[augment_result_column][0]}")

        # CSV 파일로 저장
        output_path = "../augmented_data/spell_augmented_data.csv"
        df_back_translation.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
