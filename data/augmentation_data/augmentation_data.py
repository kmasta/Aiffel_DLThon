import pandas as pd
from koeda import EDA, RD, RI, SR, RS
import asyncio
from tqdm import tqdm

# 모든 증강 기법의 인스턴스 생성
eda_instance = EDA()
rd_instance = RD()
ri_instance = RI()
sr_instance = SR()
rs_instance = RS()

# 증강 기법들을 딕셔너리로 관리
augmentation_methods = {
    "eda": eda_instance,
    "rd": rd_instance,
    "ri": ri_instance,
    "sr": sr_instance,
    "rs": rs_instance,
}


async def process_text(text, method_name):
    """텍스트 하나를 지정된 증강 방법으로 처리하는 비동기 함수"""
    method = augmentation_methods[method_name]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, method, text)


async def process_batch(texts, method_name, pbar):
    """배치 단위로 텍스트 처리"""
    tasks = [process_text(text, method_name) for text in texts]
    results = await asyncio.gather(*tasks)
    pbar.update(len(texts))  # 진행 상황 업데이트
    return results


async def process_with_all_methods(batch_size=5):
    # CSV 파일 불러오기 및 초기 데이터 확인
    df = pd.read_csv("../original_data/numeric_train.csv")
    print(f"데이터셋 크기: {df.shape}")
    print("데이터프레임 컬럼:", df.columns.tolist())

    augment_target_column = "conversation"
    augment_result_column = "augmented_conversation"

    texts = df[augment_target_column].tolist()
    augmented_results = {}

    # 각 증강 방법에 대해 데이터프레임 복사 및 처리
    for method_name in augmentation_methods.keys():
        print(f"\n{method_name.upper()} 증강 처리 시작...")
        df_augmented = df.copy()

        # 전체 텍스트 수에 대한 진행 상황 표시를 위한 tqdm 객체 생성
        with tqdm(
            total=len(texts), desc=f"{method_name.upper()} 텍스트 증강 중"
        ) as pbar:
            augmented_texts = []

            # 배치 단위로 처리
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_results = await process_batch(batch_texts, method_name, pbar)
                augmented_texts.extend(batch_results)

        # 결과를 데이터프레임에 추가
        df_augmented[augment_result_column] = augmented_texts
        augmented_results[method_name] = df_augmented

        # 결과 확인
        print(f"{method_name.upper()} 원본 텍스트 예시: {df[augment_target_column][0]}")
        print(
            f"{method_name.upper()} 증강된 텍스트 예시: {df_augmented[augment_result_column][0]}"
        )

        # CSV 파일로 저장
        df_augmented.to_csv(
            f"../augmented_data/{method_name}_augmented_data.csv", index=False
        )
        print(
            f"{method_name.upper()} 증강 데이터 저장 완료: ../augmented_data/{method_name}_augmented_data.csv"
        )


# 메인 실행 함수
if __name__ == "__main__":
    # 배치 크기 5로 설정하여 실행
    asyncio.run(process_with_all_methods(batch_size=20))
