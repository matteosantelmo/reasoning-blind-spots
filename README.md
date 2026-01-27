# Reasoning Blind Spots Benchmark

> **Reasoning Blind Spots** is a benchmark designed to stress test the reasoning capabilities of frontier AI models on tasks that are straightforward for humans but difficult for AI.
The questions are crafted to highlight the limitations of current AI systems and understand where they struggle in reasoning tasks. Download the dataset from [HuggingFace](https://huggingface.co/datasets/matsant01/blind-spots-bench) 🤗.

This benchmark was developed as part of the _"Reasoning in AI"_ course at EPFL ([MATH-700](https://edu.epfl.ch/coursebook/en/reasoning-in-artificial-intelligence-MATH-700)).
Our codebase relies on [Inspect AI](https://inspect.aisi.org.uk) as the evaluation framework.

### 🏆 Leaderboard
<!-- LEADERBOARD-START -->

#### 🖼️→📝 Multimodal-to-text Evaluation
| Model                              | Pass@1   | Pass@2   | Price / 100 Sample   |   Output Tks / Sample |
|:-----------------------------------|:---------|:---------|:---------------------|----------------------:|
| gemini-3-pro-preview               | 64.63%   | 70.73%   | $4.688               |                3722.9 |
| gemini-3-flash-preview             | 62.20%   | 73.17%   | $0.856               |                2670.5 |
| gpt-5.2                            | 46.34%   | 53.66%   | $1.263               |                 820   |
| gpt-5-mini                         | 40.24%   | 46.34%   | $0.253               |                1185   |
| Qwen3-VL-235B-A22B-Thinking        | 39.02%   | 43.90%   | $0.214               |                2761.7 |
| gemini-2.5-pro                     | 34.15%   | 41.46%   | $2.484               |                2449.7 |
| gpt-5                              | 34.15%   | 39.02%   | $2.162               |                2094.9 |
| gemini-2.5-flash                   | 34.15%   | 39.02%   | $0.888               |                3520   |
| o3                                 | 29.27%   | 31.71%   | $1.906               |                2239.1 |
| Qwen3-VL-30B-A3B-Thinking          | 24.39%   | 26.83%   | $0.046               |                2869.2 |
| Qwen3-VL-30B-A3B-Instruct          | 23.17%   | 26.83%   | $0.009               |                 447   |
| gpt-4.1                            | 21.95%   | 31.71%   | $0.259               |                 161.7 |
| Qwen3-VL-235B-A22B-Instruct        | 20.73%   | 24.39%   | $0.064               |                 694.5 |
| Mistral-Large-3-675B-Instruct-2512 | 7.32%    | 9.76%    | $0.072               |                 244.4 |

#### 📝→📝 Text-only Evaluation
| Model                              | Pass@1   | Pass@2   | Price / 100 Sample   |   Output Tks / Sample |
|:-----------------------------------|:---------|:---------|:---------------------|----------------------:|
| gemini-3-flash-preview             | 78.70%   | 85.22%   | $2.038               |                6780.9 |
| gemini-3-pro-preview               | 73.04%   | 77.39%   | $4.857               |                4033.4 |
| gpt-5.2                            | 72.17%   | 80.00%   | $2.446               |                1736.6 |
| gpt-5                              | 68.26%   | 75.65%   | $3.132               |                3121.7 |
| o3                                 | 65.22%   | 74.78%   | $2.643               |                3283.1 |
| gpt-5-mini                         | 63.48%   | 71.30%   | $0.442               |                2199.8 |
| DeepSeek-V3.2-Speciale             | 59.57%   | 64.35%   | $1.183               |                6389.8 |
| Qwen3-Next-80B-A3B-Thinking        | 58.26%   | 66.09%   | $0.243               |                7534.2 |
| gemini-2.5-pro                     | 56.96%   | 66.09%   | $3.308               |                3297.2 |
| Qwen3-VL-235B-A22B-Thinking        | 56.09%   | 64.35%   | $0.308               |                4195   |
| gpt-oss-120b                       | 53.48%   | 63.48%   | $0.018               |                1537.4 |
| Qwen3-Next-80B-A3B-Instruct        | 51.30%   | 60.00%   | $0.090               |                2774.2 |
| gemini-2.5-flash                   | 49.57%   | 59.13%   | $1.131               |                4513.9 |
| DeepSeek-V3.2                      | 49.57%   | 58.26%   | $0.186               |                 980.5 |
| gpt-oss-20b                        | 46.52%   | 58.26%   | $0.016               |                3317.4 |
| Qwen3-VL-30B-A3B-Thinking          | 42.61%   | 50.43%   | $0.060               |                4011   |
| Qwen3-VL-235B-A22B-Instruct        | 42.61%   | 52.17%   | $0.104               |                1395.7 |
| gpt-4.1                            | 41.74%   | 53.04%   | $0.509               |                 615.2 |
| Qwen3-VL-30B-A3B-Instruct          | 36.09%   | 45.22%   | $0.026               |                1683.2 |
| Mistral-Large-3-675B-Instruct-2512 | 35.65%   | 42.61%   | $0.159               |                1049.9 |
| Llama-4-Maverick-17B-128E-Instruct | 35.65%   | 41.74%   | $0.030               |                 483.5 |
| Llama-3.3-70B-Instruct             | 27.39%   | 31.30%   | $0.016               |                 274.7 |
| Apertus-70B-Instruct-2509          | 15.22%   | 20.87%   | $0.060               |                 940.5 |

<!-- LEADERBOARD-END -->


🔃 **Reproducibility:** all the results of our evaluations are saved under [`outputs/`](outputs/) and can be reproduced by running the evaluation scripts in [`scripts/`](scripts/).

---

### 🧐 Pipeline Validation
To ensure the reliability of the automated grading pipeline (relying on `gemini-3-flash` with code execution capabilities and access to ground truth annotation), we validated its performance on manually annotated question-answer pairs that we refer to as the [pipeline validation set](data/validation/). Below are the results of this validation.

##### Pipeline Validation Results
| Question Type       | Sample Size | Class Balance               | Accuracy | Precision | Recall | FPR  |
|---------------------|-------------|-----------------------------|----------|-----------|--------|------|
| multi/text-to-text  | 59          | 66% correct - 34% incorrect | 0.983    | 0.975     | 1.00   | 0.05 |
| multi/text-to-image | 8           | 50% correct - 50% incorrect | 1.00     | 1.00      | 1.00   | 0.00 |


The outputs produced by the grader during this validation can be found in [`outputs/validation/`](outputs/validation/). To reproduce the validation pipeline, you can run the [`validate_pipeline.sh`](scripts/validate_pipeline.sh) script.

---

### 🗂️ Codebase Structure
-   `conf/`: Configuration files (Hydra).
-  `notebooks/`: Jupyter notebooks for analysis and visualization.
-  `scripts/`: Scripts for reproducing evaluations.
-  `outputs/`: Outputs from evaluations and experiments.
-   `src/reasoning_blind_spots/`: Source code package.
    -   `dataset.py`: Dataset loading logic.
    -   `grader.py`: Scorer/Grader/Verifier logic for text outputs.
    -   `solver.py`: Solver/Generator logic for text outputs.
    -   `image_solver.py`: Custom solver for image generation tasks (OpenAI/Google).
    -   `image_grader.py`: Scorer/Grader for image generation outputs.
    -   `task.py`: Inspect AI task definition.
-   [`main.py`](main.py): Entry point for running the benchmark with Inspect AI.
-   [`grader_validation.py`](grader_validation.py): Script to validate the grader's performance on a subset of human-labeled data.


### 🛠️ Development
If you want to contribute to the codebase or modify it, you can follow the instructions below to set up your development environment and run the evaluation pipeline.

<details>
<summary> Environment Setup </summary>
To set up the environment for this project, follow these steps:

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e ".[dev]"
    ```
    (This installs the project in editable mode along with dependencies from `pyproject.toml` and development tools)

3.  **Setup Pre-commit Hooks:**
    Initialize the pre-commit hooks to ensure code quality checks run before every commit:
    ```bash
    pre-commit install
    ```

4.  **Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    OPENAI_API_KEY=...
    GOOGLE_API_KEY=...
    ```
</details>

<details>
<summary> Running the Benchmark </summary>

To **run the benchmark** using Inspect AI you can execute the [`main.py`](./main.py) script with your chosen configuration. Default configurations are provided in [`config.yaml`](./conf/config.yaml) or [`local_vllm.yaml`](./conf/local_vllm.yaml), but parameters can be overridden via command line arguments.

Inspect AI supports various backends (OpenAI, Anthropic, Google, local models via vLLM, etc.).
For our experiments we will mainly use Gemini/OpenAI models or open-weights models. For the latter we're using [RCP AIAAS](https://www.epfl.ch/research/facilities/rcp/ai-inference-as-a-service/) service, which hosts several open models on EPFL RCP cluster, with an API compatible with OpenAI.


Example command to run the benchmark with a specific model:
```bash
# Default config with overrides
python main.py \
    solver.model_name="gemini-2.0-flash-lite" \
    solver.backend="google" \
    solver.generate_config.reasoning_tokens=0
# Or, call the main script using a specific config file
python main.py --config-name local_vllm
```

**Image Generation Tasks:**
The benchmark also supports image generation tasks (text-to-image, image-gen). To run image generation evaluations:
```bash
# Using OpenAI GPT-Image
python main.py --config-name image_gen \
    solver.model_name="gpt-image-1" \
    solver.backend="openai"

# Using Google Imagen
python main.py --config-name image_gen \
    solver.model_name="gemini-2.5-flash-image" \
    solver.backend="google"
```

**NOTE**, to use RCP AIAAS:
- make sure to set the right values of OPENAI_BASE_URL and OPENAI_API_KEY environment variables in your `.env` file.
- you might need to be on EPFL network (or use a VPN) to access RCP services.

**Solver-only mode (skip grading):**
To run the benchmark without grading (useful for collecting solver outputs), set `grader.enabled: false` in your config:
```yaml
grader:
  enabled: false
```
</details>

<details>
<summary> Inspecting Samples and Results </summary>
Inspect AI provides a built-in dashboard that allows to visualize the results of the benchmark runs, what was generated by both the solver and the grader, and various metrics (e.g. accuracy, token counts, ...)

To **visualize the results**, you can launch Inspect AI visualization tool:
```bash
inspect view --log-dir ./outputs/<your-run-dir>/
```
Inspect AI also provides a VSCode extension that allows to visualize the results directly within VSCode.

</details>
<details>
<summary> Analysis </summary>

Using the notebooks in the `notebooks/` folder, you can analyze the results by loading the `.eval` files generated by Inspect AI runs. For example, you can use the `cost_tracking.ipynb` notebook to analyze the cost of different model runs based on token usage.
</details>
