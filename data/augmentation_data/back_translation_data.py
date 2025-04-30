import pandas as pd
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("../../.env")
CLIENT = AsyncOpenAI()


async def back_translation(text: str) -> str:
    # 영어로 번역
    en_response = await CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "당신은 전문 번역가입니다. 다음 텍스트를 영어로 번역해주세요.",
            },
            {"role": "user", "content": text},
        ],
    )
    translated_text = en_response.choices[0].message.content

    # 한국어로 다시 번역
    ko_response = await CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "당신은 전문 번역가입니다. 다음 텍스트를 한국어로 번역해주세요.",
            },
            {"role": "user", "content": translated_text},
        ],
    )
    back_translated_text = ko_response.choices[0].message.content

    return back_translated_text


async def process_batch(texts, pbar):
    """배치 단위로 텍스트 처리"""
    tasks = [back_translation(text) for text in texts]
    results = await asyncio.gather(*tasks)
    pbar.update(len(texts))  # 진행 상황 업데이트
    return results


async def process_all_texts(batch_size=5):
    df = pd.read_csv("../original_data/numeric_train.csv")
    print(f"데이터셋 크기: {df.shape}")
    print("데이터프레임 컬럼:", df.columns.tolist())
    augment_target_column = "conversation"
    augment_result_column = "augmented_conversation"

    df_back_translation = df.copy()
    texts = df[augment_target_column].tolist()

    # 전체 텍스트 수에 대한 진행 상황 표시를 위한 tqdm 객체 생성
    with tqdm(total=len(texts), desc="텍스트 처리 중") as pbar:
        augmented_texts = []

        # 배치 단위로 처리
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = await process_batch(batch_texts, pbar)
            augmented_texts.extend(batch_results)

    # 결과를 데이터프레임에 추가
    df_back_translation[augment_result_column] = augmented_texts

    # 결과 확인
    print(f"원본 텍스트 예시: {df[augment_target_column][0]}")
    print(f"증강된 텍스트 예시: {df_back_translation[augment_result_column][0]}")

    # CSV 파일로 저장
    output_path = "../augmented_data/back_translation_augmented_data.csv"
    df_back_translation.to_csv(output_path, index=False)


# 메인 실행 함수
if __name__ == "__main__":
    # 배치 크기 5로 설정하여 실행
    asyncio.run(process_all_texts(batch_size=20))
