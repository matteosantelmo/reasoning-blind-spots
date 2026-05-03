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

from pathlib import Path
import json


def split_jsonl(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    num_input = 0
    num_output = 0
    num_skipped = 0
    """
    Split a composed task into multiple subtask samples using failure_modes lines.

    Only one failure mode line is evaluated per split sample.
    The sample id becomes: original_id_0, original_id_1, ...
    """
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue

            num_input += 1
            record = json.loads(line)
            parent_id = record.get("index") or record.get("QID") or record.get("id")
            if parent_id is None:
                raise ValueError(f"Record missing index/QID/id: {record}")

            full_solution = record.get("solution", "")

            failure_mode_lines, subtask_lines = split_failure_modes_tasks(record.get("failure_mode", ""))
            if not failure_mode_lines:
                raise ValueError(
                    f"Record {parent_id} has multi_subcategory_flag=1 "
                    "but has no non-empty failure_modes lines."
                )


            for i, failure_mode_line in enumerate(failure_mode_lines):
                sub_record = dict(record)

                sub_id = f"{parent_id}_{i}"
                sub_record["sub-task_type"] = subtask_lines[i]
                sub_record["task_type"] = SUBTASK_TO_TASK_TAXONOMY[sub_record["sub-task_type"]]
                sub_record["index"] = sub_id
                sub_record["solution"] = (
                    "This is a composed task.\n\n"
                    f"The full reference solution is:\n{full_solution}\n\n"
                    "The full solution may contain multiple requirements, but this evaluation is only for one aspect.\n\n"
                    f"We provide the specific aspect to evaluate below:\n{failure_mode_line}\n\n"
                    "Judge the generated image only according to this specific aspect.\n"
                    "Do not require all conditions in the full solution to be satisfied.\n"
                    "Do not penalize unrelated mistakes from other aspects.\n"
                    "If this specific aspect is satisfied, grade it as correct.\n"
                    "If this specific aspect is violated, missing, ambiguous, occluded, or only implied, grade it as incorrect."
                )
                fout.write(json.dumps(sub_record, ensure_ascii=False) + "\n")
                num_output += 1
    print(f"Input records: {num_input}")
    print(f"Skipped non-multi records: {num_skipped}")
    print(f"Output split records: {num_output}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    split_jsonl(
        input_path="../data/multicategory/multi_category_grader_inputs_reeval.jsonl",
        output_path="../data/multicategory/multi_category_grader_inputs_reevalsplit.jsonl",
    )