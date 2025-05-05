import subprocess

# 전처리 대상 설정 목록
configs = [
    # {
    #     "input": "../../data/processed/spell_checked/clsp_test.csv",
    #     "output": "../../data/processed/stopwords/st_test.csv"
    # },
    # {
    #     "input": "../../data/processed/spell_checked/generated/clsp_bad.csv",
    #     "output": "../../data/processed/stopwords/generated/st_bad.csv"
    # },
    # {
    #     "input": "../../data/processed/spell_checked/generated/clsp_general.csv",
    #     "output": "../../data/processed/stopwords/generated/st_general.csv"
    # },
    # {
    #     "input": "../../data/processed/spell_checked/generated/clsp_general2.csv",
    #     "output": "../../data/processed/stopwords/generated/st_general2.csv"
    # },
    # {
    #     "input": "../../data/processed/spell_checked/aihub/clsp_sns_multiturn.csv",
    #     "output": "../../data/processed/stopwords/aihub/st_sns_multiturn.csv"
    # },
    {
        "input": "../../data/processed/spell_checked/augmented/clsp_backT.csv",
        "output": "../../data/processed/stopwords/augmented/st_backT.csv"
    },
    {
        "input": "../../data/processed/spell_checked/augmented/clsp_eda.csv",
        "output": "../../data/processed/stopwords/augmented/st_eda.csv"
    },
    {
        "input": "../../data/processed/spell_checked/augmented/clsp_rd.csv",
        "output": "../../data/processed/stopwords/augmented/st_rd.csv"
    },
    {
        "input": "../../data/processed/spell_checked/augmented/clsp_ri.csv",
        "output": "../../data/processed/stopwords/augmented/st_ri.csv"
    },
    {
        "input": "../../data/processed/spell_checked/augmented/clsp_rs.csv",
        "output": "../../data/processed/stopwords/augmented/st_rs.csv"
    },
    {
        "input": "../../data/processed/spell_checked/augmented/clsp_sr.csv",
        "output": "../../data/processed/stopwords/augmented/st_sr.csv"
    },
    # 더 많은 파일을 추가할 수 있습니다.
]

# stopwords.py 경로
SCRIPT_PATH = "stopwords.py"

# 실행 루프
for cfg in configs:
    cmd = [
        "python", SCRIPT_PATH,
        "--input", cfg["input"],
        "--output", cfg["output"],
    ]
    if "text_col" in cfg:
        cmd += ["--text_col", cfg["text_col"]]
    if "label_col" in cfg:
        cmd += ["--label_col", cfg["label_col"]]
    print(f"🔧 Running: {' '.join(cmd)}")
    subprocess.run(cmd)
