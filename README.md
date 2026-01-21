# Reasoning Blind Spots Benchmark

> **Reasoning Blind Spots** is a benchmark designed to stress test the reasoning capabilities of frontier AI models on tasks that are straightforward for humans but difficult for AI.
The questions are crafted to highlight the limitations of current AI systems and understand where they struggle in reasoning tasks. Download the dataset from [HuggingFace](https://huggingface.co/datasets/matsant01/blind-spots-bench) 🤗.

This benchmark was developed as part of the _"Reasoning in AI"_ course at EPFL ([MATH-700](https://edu.epfl.ch/coursebook/en/reasoning-in-artificial-intelligence-MATH-700)).
Our codebase relies on [Inspect AI](https://inspect.aisi.org.uk) as the evaluation framework.

### Leaderboard
<!-- LEADERBOARD-START -->

#### Multimodal-to-text Evaluation
| Model                              | Accuracy   | Price / 100 Sample   |   Output Tks / Sample |
|:-----------------------------------|:-----------|:---------------------|----------------------:|
| gemini-3-flash-preview             | 64.86%     | $0.710               |                2182.7 |
| gpt-5.2                            | 45.95%     | $1.310               |                 857.3 |
| gpt-5-mini                         | 37.84%     | $0.284               |                1343.8 |
| Qwen3-VL-235B-A22B-Thinking        | 35.14%     | $0.237               |                3076.1 |
| gpt-5                              | 35.14%     | $1.941               |                1803.1 |
| gemini-2.5-flash                   | 32.43%     | $0.760               |                3007.9 |
| gemini-2.5-pro                     | 32.43%     | $3.234               |                3199.5 |
| o3                                 | 32.43%     | $1.530               |                1772.6 |
| Qwen3-VL-235B-A22B-Instruct        | 29.73%     | $0.085               |                 998.8 |
| Qwen3-VL-30B-A3B-Thinking          | 27.03%     | $0.047               |                2992.8 |
| Qwen3-VL-30B-A3B-Instruct          | 24.32%     | $0.010               |                 472.5 |
| gpt-4.1                            | 18.92%     | $0.255               |                 160.2 |
| Mistral-Large-3-675B-Instruct-2512 | 5.41%      | $0.069               |                 232.3 |

#### Text-only Evaluation
| Model                              | Accuracy   | Price / 100 Sample   |   Output Tks / Sample |
|:-----------------------------------|:-----------|:---------------------|----------------------:|
| gemini-3-flash-preview             | 78.95%     | $2.121               |                7057   |
| gpt-5                              | 70.18%     | $3.235               |                3224.7 |
| o3                                 | 68.42%     | $3.055               |                3798.4 |
| gpt-5.2                            | 68.42%     | $2.404               |                1706.4 |
| gpt-5-mini                         | 64.04%     | $0.420               |                2090.7 |
| gemini-2.5-pro                     | 57.02%     | $2.941               |                2930.7 |
| gpt-oss-120b                       | 56.14%     | $0.020               |                1687.2 |
| Qwen3-VL-235B-A22B-Thinking        | 55.26%     | $0.308               |                4192.5 |
| DeepSeek-V3.2-Speciale             | 53.51%     | $1.141               |                6165.5 |
| gemini-2.5-flash                   | 50.88%     | $0.927               |                3696.2 |
| DeepSeek-V3.2                      | 49.12%     | $0.183               |                 964.3 |
| Qwen3-VL-235B-A22B-Instruct        | 44.74%     | $0.110               |                1480.8 |
| Qwen3-VL-30B-A3B-Thinking          | 42.98%     | $0.063               |                4170.3 |
| gpt-4.1                            | 42.98%     | $0.496               |                 599.2 |
| Mistral-Large-3-675B-Instruct-2512 | 37.72%     | $0.201               |                1338.8 |
| Qwen3-VL-30B-A3B-Instruct          | 35.09%     | $0.024               |                1591.4 |
| Apertus-70B-Instruct-2509          | 16.67%     | $0.057               |                 895   |

<!-- LEADERBOARD-END -->

---

### Codebase Structure
-   `conf/`: Configuration files (Hydra).
-  `notebooks/`: Jupyter notebooks for analysis and visualization.
-  `scripts/`: Scripts for reproducing evaluations.
-  `outputs/`: Outputs from evaluations and experiments.
-   `src/reasoning_blind_spots/`: Source code package.
    -   `dataset.py`: Dataset loading logic.
    -   `grader.py`: Scorer/Grader/Verifier logic.
    -  `solver.py`: Solver/Generator logic.
    -   `task.py`: Inspect AI task definition.
-   [`main.py`](main.py): Entry point for running the benchmark with Inspect AI.
-   [`grader_validation.py`](grader_validation.py): Script to validate the grader's performance on a subset of human-labeled data.


### Development
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
    ANTHROPIC_API_KEY=...
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

**NOTE**, to use RCP AIAAS:
- make sure to set the right values of OPENAI_BASE_URL and OPENAI_API_KEY environment variables in your `.env` file.
- you might need to be on EPFL network (or use a VPN) to access RCP services.
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
