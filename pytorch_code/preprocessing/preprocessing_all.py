import subprocess

# 전처리 대상 설정 목록
configs = [
    {
        "input": "../../data/processed/clean_test.csv",
        "output": "../../data/processed/spell_checked/clsp_test.csv",
        "text_col": "clean_text"
    },
    {
        "input": "../../data/processed/augmented/clean_backT.csv",
        "output": "../../data/processed/spell_checked/augmented/clsp_backT.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/augmented/clean_eda.csv",
        "output": "../../data/processed/spell_checked/augmented/clsp_eda.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/augmented/clean_rd.csv",
        "output": "../../data/processed/spell_checked/augmented/clsp_rd.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/augmented/clean_ri.csv",
        "output": "../../data/processed/spell_checked/augmented/clsp_ri.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/augmented/clean_rs.csv",
        "output": "../../data/processed/spell_checked/augmented/clsp_rs.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/augmented/clean_sr.csv",
        "output": "../../data/processed/spell_checked/augmented/clsp_sr.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/generated/clean_bad.csv",
        "output": "../../data/processed/spell_checked/generated/clsp_bad.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/generated/clean_general.csv",
        "output": "../../data/processed/spell_checked/generated/clsp_general.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/daily_data/daily_train2.csv",
        "output": "../../data/processed/spell_checked/generated/clsp_general2.csv",
        "text_col": "conversation",
        "label_col": "class",
        "skip_cleaning": "true"
    },
    {
        "input": "../../data/processed/original/clean_train.csv",
        "output": "../../data/processed/spell_checked/original/clsp_train.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    {
        "input": "../../data/processed/aihub/clean_sns_multiturn.csv",
        "output": "../../data/processed/spell_checked/aihub/clsp_sns_multiturn.csv",
        "text_col": "clean_text",
        "label_col": "label"
    },
    # 필요 시 추가
]

# 공통 실행 경로
SCRIPT_PATH = "preprocessing.py"

# 실행 루프
for cfg in configs:
    cmd = [
        "python", SCRIPT_PATH,
        "--inputs", cfg["input"],
        "--output", cfg["output"],
    ]
    if "text_col" in cfg:
        cmd += ["--text_col", cfg["text_col"]]
    if "label_col" in cfg:
        cmd += ["--label_col", cfg["label_col"]]
    if "spellchecking" in cfg:
        cmd += ["--spellchecking", cfg["spellchecking"]]
    if "skip_cleaning" in cfg:
        cmd += ["--skip_cleaning", cfg["skip_cleaning"]]
    print(f"🔧 Running: {' '.join(cmd)}")
    subprocess.run(cmd)
