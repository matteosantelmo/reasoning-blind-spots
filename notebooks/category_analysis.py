import pandas as pd

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


def load_task_taxonomy(dataset_path, sheet_name):
    df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    print(df.head())
    print(df.columns)

    qid_col = "QID"
    task_col = "task_type"
    subtask_col = "sub-task type"
    taxonomy_dict = {}
    for _, row in df.iterrows():
        qid = row[qid_col]
        if row["multi_subcategory_flag"] == 1:
            failure_mode_lines, subtask_lines = split_failure_modes_tasks(row["failure_mode"])
            for i in range(failure_mode_lines):
                sub_id = f"qid_{i}"
                sub_task_type = subtask_lines[i]
                task_type = SUBTASK_TO_TASK_TAXONOMY[sub_task_type]
                taxonomy_dict[sub_id] = {
                    "task_type": task_type,
                    "sub-task type": sub_task_type,
                }
        else:
            

            taxonomy_dict[qid] = {
                "task_type": row[task_col],
                "sub-task type": row[subtask_col],
            }

    return taxonomy_dict


taxonomy_dict = load_task_taxonomy("../data/reasoning_course.xlsx", "dataset-0319")
print(taxonomy_dict)
# # load processed_results_all.csv
# csv_path = "../data/processed_results_all.csv"
# # for a certain model, we get its results on ""
# df = pd.read_csv(csv_path)
# print(df.head())
# print(df.columns)
