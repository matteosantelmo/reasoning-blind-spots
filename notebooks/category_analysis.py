import pandas as pd
import json
from pathlib import Path


SUBTASK_TAXONOMY = [
    "attribute and pattern recognition",
    "spatial reasoning",
    "generative counting",
    "perceptual counting",
    "attribute binding",
    "logical reasoning",
    "arithmetic reasoning",
    "geometric and graph reasoning",
    "constraint reasoning",
    "irrelevant-context robustness",
    "character-level manipulation",
    "world knowledge",
]
TASK_TAXONOMY = [
    "object-centric",
    "abstract reasoning",
    "language and knowledge",
]

SUBTASK_TO_TASK_TAXONOMY = {
    "attribute and pattern recognition": "object-centric",
    "spatial reasoning": "object-centric",
    "generative counting": "object-centric",
    "perceptual counting": "object-centric",
    "attribute binding": "object-centric",

    "logical reasoning": "abstract reasoning",
    "arithmetic reasoning": "abstract reasoning",
    "geometric and graph reasoning": "abstract reasoning",
    "constraint reasoning": "abstract reasoning",

    "irrelevant-context robustness": "language and knowledge",
    "character-level manipulation": "language and knowledge",
    "world knowledge": "language and knowledge",
}



def split_failure_modes_tasks(failure_modes) -> tuple[list[str], list[str]]:

    if failure_modes is None:
        return [], []
    failure_lines = [line.strip() for line in str(failure_modes).splitlines() if line.strip()]
    # Each line is expected to contain exactly one subtask taxonomy.
    match_lines = []
    for line in failure_lines:
        line = line.lower()
        matches = [taxonomy for taxonomy in SUBTASK_TAXONOMY if taxonomy.lower() in line]
        if len(matches) == 0:
            raise ValueError(f"Could not find any subtask taxonomy in failure mode line: {line}")

        if len(matches) > 1:
            raise ValueError(f"Found multiple subtask taxonomies in failure mode line: {line}." f"Matches: {matches}")
        match_lines.append(matches[0])
    return failure_lines, match_lines


def read_jsonl_safely(input_jsonl: str) -> pd.DataFrame:
    path = Path(input_jsonl)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Bad JSON at line {line_no}: {e}")
                print(line[:500])
                raise

    if not records:
        raise ValueError(f"No valid records found in {path.resolve()}")

    return pd.DataFrame(records)


def load_task_taxonomy(dataset_path, sheet_name):
    df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    qid_col = "QID"
    task_col = "task_type"
    subtask_col = "sub-task type"
    taxonomy_dict = {}
    for _, row in df.iterrows():
        if row["valid_flag"] != 1 or pd.isna(row[qid_col]):
            continue
        qid = row[qid_col]
        if isinstance(qid, float) and qid.is_integer():
            qid = str(int(qid))
        else:
            qid = str(qid).strip()
        if row["multi_subcategory_flag"] == 1:
            failure_mode_lines, subtask_lines = split_failure_modes_tasks(row["Failure mode"])
            for i in range(len(failure_mode_lines)):
                sub_id = f"{qid}_{i}"
                sub_task_type = subtask_lines[i]
                task_type = SUBTASK_TO_TASK_TAXONOMY[sub_task_type]
                taxonomy_dict[sub_id] = {
                    "task_type": task_type,
                    "sub-task type": sub_task_type,
                    "question_type": row["question_type"],
                }
        else:
            

            taxonomy_dict[qid] = {
                "task_type": row[task_col],
                "sub-task type": row[subtask_col],
                "question_type": row["question_type"],
            }

    return taxonomy_dict



def jsonl_to_csv(input_jsonl: str, output_csv: str, taxonomy_dict: str) -> None:
    def extract_model_name(solver_answer: str) -> str:
        IMAGE_GEN_MODEL_NAMES = [
            "gpt-image-1-mini",
            "gemini-2.5-flash-image",
            "gpt-image-1.5",
            "gpt-image-2",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
        ]
        if not isinstance(solver_answer, str):
            return ""

        for model_name in IMAGE_GEN_MODEL_NAMES:
            if model_name in solver_answer:
                return model_name
        return ""

    df = read_jsonl_safely(input_jsonl)

    index_str = df["index"].astype(str)

    out = pd.DataFrame({
        "qid": index_str,
        "model_name": df["solver_answer"].apply(extract_model_name),
        "task_type": index_str.map(
            lambda qid: taxonomy_dict.get(qid, {}).get("task_type", "")
        ),
        "sub-task type": index_str.map(
            lambda qid: taxonomy_dict.get(qid, {}).get("sub-task type", "")
        ),
        "score": df["grader_score"].map({"C": 1, "I": 0}),
    })

    out.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


taxonomy_dict = load_task_taxonomy("../data/reasoning_course.xlsx", "dataset-0319")
jsonl_to_csv(input_jsonl="../outputs/multicategory/validation_results.jsonl", output_csv="../data/multicategory_scores.csv", taxonomy_dict=taxonomy_dict)



def analysis(taxonomy_dict):
    results_all = "../data/processed_results_all.csv"
    results_multi = "../data/multicategory_scores.csv"
    df_all = pd.read_csv(results_all)
    df_multi = pd.read_csv(results_multi)
    model_score = {}
    model_total = {}
    all_taxonomy = SUBTASK_TAXONOMY + TASK_TAXONOMY
    hard_total = {
        "LM": {key: 0 for key in all_taxonomy},
        "VLM": {key: 0 for key in all_taxonomy},
        "GEN": {key: 0 for key in all_taxonomy},
        "ALL": {key: 0 for key in all_taxonomy},
    }
    for k, v in taxonomy_dict.items():
        if v["question_type"] == "text-only":
            hard_total["LM"][v["task_type"].lower()] += 1
            hard_total["LM"][v["sub-task type"].lower()] += 1
        if v["question_type"] in ["text-to-image", "text-to-multi", "multi-to-image"]:
            hard_total["GEN"][v["task_type"].lower()] += 1
            hard_total["GEN"][v["sub-task type"].lower()] += 1
        if v["question_type"] in ["text-only", "multi-to-text"]:
            hard_total["VLM"][v["task_type"].lower()] += 1
            hard_total["VLM"][v["sub-task type"].lower()] += 1
        hard_total["ALL"][v["task_type"].lower()] += 1
        hard_total["ALL"][v["sub-task type"].lower()] += 1
    def cac_score(qid, model_name, score):
        task_type = taxonomy_dict[qid]["task_type"].lower()
        subtask_type = taxonomy_dict[qid]["sub-task type"].lower()
        model_score[model_name][task_type] += score
        model_score[model_name][subtask_type] += score
        model_total[model_name][task_type] += 1
        model_total[model_name][subtask_type] += 1

    for _, row in df_all.iterrows():
        qid = str(row["qid"]) if row["qid"] < 1000 else str(int(row["qid"] - 1000))
        model_name = row["model_name"]
        if model_name not in model_score:
            model_score[model_name] = {key: 0 for key in all_taxonomy}
            model_total[model_name] = {key: 0 for key in all_taxonomy}
        score = row["score"]
        if qid not in taxonomy_dict:
            assert row["model_type"] not in ["LM", "VLM"] 
            matched = df_multi[
                df_multi["qid"].astype(str).str.match(rf"^{qid}_\d+$")
                & (df_multi["model_name"] == model_name)
            ]
            matched_result = 1
            for _, mrow in matched.iterrows():
                matched_result &= mrow["score"]
                cac_score(str(mrow["qid"]), model_name, mrow["score"])
            if matched_result != score:
                print(f"Inconsistent {qid}, {model_name}, {matched_result}, {score}")
        else:
            cac_score(qid, model_name, score)

    model_type_map = df_all[["model_name", "model_type"]].drop_duplicates()
    result = {}

    for model_type, group in model_type_map.groupby("model_type"):
        model_names = group["model_name"].dropna().unique().tolist()
        cur_total = hard_total[model_type]
        for name in model_names:
            if model_type in ["LM", "VLM"]:
                result[name] = {
                    task: model_score[name].get(task, 0) / 4 / cur_total.get(task)
                    if cur_total.get(task) != 0 else None
                    for task in cur_total
                }
                for task in cur_total:
                    if cur_total[task] != model_total[name][task] / 4:
                        print(cur_total[task], model_total[name][task], task, name)
            else:
                result[name] = {
                    task: model_score[name].get(task, 0) / cur_total.get(task)
                    if cur_total.get(task) != 0 else None
                    for task in cur_total
                }
                for task in cur_total:
                    if cur_total[task] != model_total[name][task]:
                        print(cur_total[task], model_total[name][task], task, name)
            result[name]["model_type"] = model_type
    merged_results = {
        **hard_total,
        **result,
    }
    df = pd.DataFrame.from_dict(merged_results, orient="index")
    df.index.name = "key"
    df = df.reset_index()
    summary_rows = {"LM", "VLM", "GEN", "ALL"}
    for col in df.columns:
        if col != "key":
            df[col] = df.apply(
                lambda row: row[col] if row["key"] in summary_rows
                else f"{row[col] * 100:.2f}%" if pd.notna(row[col])
                else "",
                axis=1,
            )

    df.to_csv("../data/task_score_results.csv", index=False)
analysis(taxonomy_dict=taxonomy_dict)