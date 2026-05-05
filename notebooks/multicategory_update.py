import json
from pathlib import Path


def filter_and_update_failure_modes(
    input_path: str,
    output_path: str,
    new_failure_modes: dict,
    keep_indices: set[str] = {"48", "106"},
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    kept = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue

            record = json.loads(line)

            index = str(record.get("index"))

            if index not in keep_indices:
                continue

            if index in new_failure_modes:
                record["failure_mode"] = new_failure_modes[index]
            else:
                raise ValueError(f"No new failure_modes provided for index={index}")

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} records to {output_path}")


if __name__ == "__main__":
    new_failure_modes = {
        "48": "If the image shows shapes with colors different from those requested—such as a rectangle not in red, a square not in green, a triangle not in purple, or a parallelogram not in blue—then it fails in attribute binding.\nIf the image places shapes in positions different from those requested—such as the red rectangle not being to the left of the green square, the purple triangle not being above the green square, the green square not being above the blue parallelogram—then it fails in spatial reasoning.\nIf the image does not show two triangles—a purple one above the green square and another below it—or states that the constraints cannot be satisfied, then it fails in constraint reasoning.",
        "106": "If the image shows a number of black boxes other than two, or a number of small green spheres other than five, then the model fails in generative counting.\nIf the image does not satisfy the required spatial relationships between objects—such as the black boxes not being in front of the green spheres, the red pyramid not being to the right of the green spheres, or the small green spheres not being at the bottom of the image—then the model fails in spatial reasoning.\nIf the image does not satisfy the required properties of each object—such as spheres not being green, boxes not being black with white circles, or the absence of a red pyramid—then the model fails in attribute binding.",
    }

    filter_and_update_failure_modes(
        input_path="../data/multicategory/multi_category_grader_inputs.jsonl",
        output_path="../data/multicategory/multi_category_grader_inputs_reeval.jsonl",
        new_failure_modes=new_failure_modes,
    )