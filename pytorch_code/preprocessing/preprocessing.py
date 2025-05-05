"""
모델 사용 예시(커널에서)
python preprocessing.py \
  --inputs ../../data/augmented_data/spell_augmented_data.csv \
  --output ../../data/processed/augmented/clean_spell.csv \
  --text_col augmented_conversation \
  --label_col class \
  --spellchecking true \
  --skip_cleaning true
주의: skip_cleaning과 spellchecking의 default는 true
"""

import os
import re
import pandas as pd
import argparse
from hanspell import spell_checker

label_to_id = {
    "협박 대화": 0,
    "갈취 대화": 1,
    "직장 내 괴롭힘 대화": 2,
    "기타 괴롭힘 대화": 3,
    "일반 대화": 4
}

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^\w\s가-힣.,!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def correct_spelling(text: str) -> str:
    try:
        result = spell_checker.check(text.replace("\n", " "))
        if not result.result or not result.checked.strip():
            return text  # 결과가 없거나 빈 문자열이면 원본 반환
        return result.checked
    except Exception:
        return text

def preprocess_and_merge(input_csv_paths, output_csv_path, text_col_name, label_col_name=None, use_spellchecking=True, skip_cleaning=True):
    dfs = []

    for path in input_csv_paths:
        print(f"Loading: {path}")
        df = pd.read_csv(path)

        if text_col_name not in df.columns:
            raise ValueError(f"{path} does not contain the specified text column: '{text_col_name}'")

        if not skip_cleaning:
            texts = df[text_col_name].astype(str).apply(clean_text)
        else:
            texts = df[text_col_name].astype(str)

        if use_spellchecking:
            print("✔ 맞춤법 검사 적용 중...")
            texts = texts.apply(correct_spelling)

        df["clean_text"] = texts

        if label_col_name:
            if label_col_name not in df.columns:
                raise ValueError(f"{path} does not contain the specified label column: '{label_col_name}'")

            label_values = df[label_col_name]

            if pd.api.types.is_integer_dtype(label_values):
                if not set(label_values).issubset(set(label_to_id.values())):
                    raise ValueError(f"Integer labels in {path} are out of expected range: {set(label_values) - set(label_to_id.values())}")
                df["label"] = label_values
            else:
                if not set(label_values).issubset(label_to_id.keys()):
                    raise ValueError(f"Unexpected class values in {path}: {set(label_values) - label_to_id.keys()}")
                df["label"] = label_values.map(label_to_id)

            dfs.append(df[["clean_text", "label"]])

        else:
            dfs.append(df[["clean_text"]])

    merged_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    merged_df.to_csv(output_csv_path, index=False)
    print(f"✅ 저장 완료: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and merge multiple chat CSV files")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to input CSV files")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--text_col", type=str, default="conversation", help="Column name for conversation text")
    parser.add_argument("--label_col", type=str, default=None, help="(Optional) Column name for label/class")
    parser.add_argument("--spellchecking", type=str, default="true", help="Whether to apply spelling correction (true/false)")
    parser.add_argument("--skip_cleaning", type=str, default="true", help="Skip text cleaning if already preprocessed (true/false)")

    args = parser.parse_args()
    use_spellchecking = args.spellchecking.lower() == "true"
    skip_cleaning = args.skip_cleaning.lower() == "true"

    preprocess_and_merge(args.inputs, args.output, args.text_col, args.label_col, use_spellchecking, skip_cleaning)
