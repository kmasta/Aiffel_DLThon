import os
import re
import pandas as pd
import argparse

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

def preprocess_and_merge(input_csv_paths, output_csv_path):
    dfs = []

    for path in input_csv_paths:
        print(f"Loading: {path}")
        df = pd.read_csv(path)

        if "conversation" in df.columns:
            raw_text_col = "conversation"
        elif "text" in df.columns:
            raw_text_col = "text"
        else:
            raise ValueError(f"{path} must contain either 'conversation' or 'text' column.")

        df["clean_text"] = df[raw_text_col].apply(clean_text)

        if "class" in df.columns:
            if not set(df["class"]).issubset(label_to_id.keys()):
                raise ValueError(f"Unexpected class values in {path}: {set(df['class']) - label_to_id.keys()}")
            df["label"] = df["class"].map(label_to_id)
            dfs.append(df[["clean_text", "label"]])
        else:
            dfs.append(df[["clean_text"]])

    merged_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Saved merged and preprocessed file to: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and merge multiple chat CSV files")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to input CSV files")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()

    preprocess_and_merge(args.inputs, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and merge multiple chat CSV files")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to input CSV files")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()

    preprocess_and_merge(args.inputs, args.output)
