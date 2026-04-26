# Reasoning Blind Spots Benchmark

> **Reasoning Blind Spots** is a benchmark designed to stress test the reasoning capabilities of frontier AI models on tasks that are straightforward for humans but difficult for AI.
The questions are crafted to highlight the limitations of current AI systems and understand where they struggle in reasoning tasks. Download the dataset from [HuggingFace](https://huggingface.co/datasets/matsant01/blind-spots-bench) 🤗.

This benchmark was developed as part of the _"Reasoning in AI"_ course at EPFL ([MATH-700](https://edu.epfl.ch/coursebook/en/reasoning-in-artificial-intelligence-MATH-700)).
Our codebase relies on [Inspect AI](https://inspect.aisi.org.uk) as the evaluation framework.

---

### 🏆 Leaderboard
<!-- LEADERBOARD-START -->

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
For our experiments we will mainly use Gemini/OpenAI models or open-weights models. For the latter we support OpenAI-compatible endpoints such as [RCP AIAAS](https://www.epfl.ch/research/facilities/rcp/ai-inference-as-a-service/) and Swiss AI serving on CSCS.


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

**NOTE**, to use CSCS Swiss AI serving:
- set `CSCS_SERVING_API` in your `.env` file.
- optionally set `CSCS_BASE_URL`; the default is `https://api.swissai.svc.cscs.ch/v1`.
- use the `openai-api/cscs/<vendor>/<model>` model id pattern when configuring the solver directly.

```bash
python main.py --config-name cscs
```

**Tool-enabled text evaluations:**
Inspect AI can now expose `code_execution()` and `web_search()` to solver models on text-output tasks with a bounded multi-step loop.

For tool-enabled runs, the relevant knobs are:
- `solver.tools.enabled`
- `solver.tools.code_execution`
- `solver.tools.web_search`
- `solver.tools.web_search_providers`
- `solver.tools.max_additional_messages`
- `sandbox` (required for client-side tool execution, e.g. `sandbox: docker`)

For `solver.tools.web_search`, native OpenAI and Gemini backends automatically use their internal web-search tool when no provider override is given. Self-hosted and OpenAI-compatible endpoints such as `openai-api/...`, RCP, CSCS, and local vLLM require an explicit external provider, for example `solver.tools.web_search_providers=["tavily"]`. For that, a TAVILY_API_KEY will also be needed.

The model catalog in [`data/models_pricing.csv`](data/models_pricing.csv) includes a `tool_calling` column indicating whether a model can be used with text-mode tools in this benchmark, either natively or through Inspect AI tool emulation.

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
