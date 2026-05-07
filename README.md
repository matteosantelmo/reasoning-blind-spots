# Reasoning Blind Spots Benchmark

> **`blind-spots-bench`** is a benchmark designed to stress test the reasoning capabilities of frontier AI models on tasks that are straightforward for humans but difficult for AI.
The questions are crafted to highlight the limitations of current AI systems and understand where they struggle in reasoning tasks. Download the dataset from [HuggingFace](https://huggingface.co/datasets/blind-spots-neurips/blind-spots-bench) 🤗.

Our codebase relies on [Inspect AI](https://inspect.aisi.org.uk) as the evaluation framework.

---

### Results


<div align="left">
  <img src="notebooks/assets/png/text_accuracy_vs_aa_index.png" width="500">

</div>


<details>
<summary> 🏆 Complete leaderboard </summary>
<div>
<table>
  <thead>
    <tr>
      <th>Group</th>
      <th>Model</th>
      <th>Text mean@4</th>
      <th>Text pass@4</th>
      <th>Text out-tks</th>
      <th>Text cost</th>
      <th>Multi mean@4</th>
      <th>Multi pass@4</th>
      <th>Multi out-tks</th>
      <th>Multi cost</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Text-only</td><td>GLM-4.7</td><td>61.3</td><td>75.9</td><td>5346</td><td>0.91</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>Text-only</td><td>GLM-5</td><td>64.4</td><td>77.8</td><td>3404</td><td>1.11</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>Text-only</td><td>GLM-5.1</td><td>67.4</td><td>76.9</td><td>2948</td><td>0.69</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>Text-only</td><td>DeepSeek-V4-Flash</td><td>67.6</td><td>79.6</td><td>3648</td><td>0.29</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>Text-only</td><td>DeepSeek-V4-Pro</td><td>70.4</td><td>80.6</td><td>2232</td><td>0.63</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>Text-only</td><td>GPT-oss-20b</td><td>52.3</td><td>67.6</td><td>2611</td><td>0.02</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>Text-only</td><td>GPT-oss-120b</td><td>60.2</td><td>72.2</td><td>1555</td><td>0.02</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>Text-only</td><td>Qwen3-Next-80B-A3B</td><td>59.5</td><td>72.2</td><td>6460</td><td>0.27</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td><td>&ndash;</td></tr>
    <tr><td>VLM</td><td>Kimi-K2.5</td><td>68.3</td><td>80.6</td><td>4163</td><td>0.86</td><td>42.4</td><td>53.5</td><td>4994</td><td>1.07</td></tr>
    <tr><td>VLM</td><td>Kimi-K2.6</td><td>65.0</td><td>78.7</td><td>3797</td><td>0.70</td><td>41.9</td><td>53.5</td><td>4501</td><td>0.87</td></tr>
    <tr><td>VLM</td><td>Gemma-4-E2B-it</td><td>32.4</td><td>42.6</td><td>1533</td><td>0.01</td><td>18.0</td><td>30.2</td><td>1049</td><td>0.01</td></tr>
    <tr><td>VLM</td><td>Gemma-4-E4B-it</td><td>41.4</td><td>57.4</td><td>1601</td><td>0.01</td><td>22.7</td><td>32.6</td><td>1335</td><td>0.01</td></tr>
    <tr><td>VLM</td><td>Gemma-4-26B-A4B-it</td><td>62.5</td><td>72.2</td><td>4541</td><td>0.06</td><td>40.7</td><td>53.5</td><td>3587</td><td>0.05</td></tr>
    <tr><td>VLM</td><td>Qwen3-VL-30B-A3B</td><td>49.5</td><td>66.7</td><td>3673</td><td>0.07</td><td>25.0</td><td>37.2</td><td>2796</td><td>0.06</td></tr>
    <tr><td>VLM</td><td>Qwen3-VL-235B-A22B</td><td>58.3</td><td>71.3</td><td>3435</td><td>0.32</td><td>32.0</td><td>39.5</td><td>2604</td><td>0.26</td></tr>
    <tr><td>VLM</td><td>Qwen3.5-35B-A3B</td><td>63.9</td><td>74.1</td><td>7278</td><td>0.11</td><td>47.1</td><td>69.8</td><td>6492</td><td>0.10</td></tr>
    <tr><td>VLM</td><td>Qwen3.5-122B-A10B</td><td>63.9</td><td>75.0</td><td>5637</td><td>0.25</td><td>48.8</td><td>62.8</td><td>3350</td><td>0.16</td></tr>
    <tr><td>VLM</td><td>Qwen3.5-397B-A17B</td><td>71.1</td><td>81.5</td><td>5849</td><td>0.89</td><td>48.3</td><td>58.1</td><td>5479</td><td>0.85</td></tr>
    <tr><td>VLM</td><td>Gemini-2.5-flash</td><td>55.3</td><td>70.4</td><td>3884</td><td>0.97</td><td>32.0</td><td>41.9</td><td>3383</td><td>0.85</td></tr>
    <tr><td>VLM</td><td>Gemini-2.5-pro</td><td>60.4</td><td>76.9</td><td>2905</td><td>2.92</td><td>36.0</td><td>44.2</td><td>3283</td><td>3.32</td></tr>
    <tr><td>VLM</td><td>Gemini-3-flash</td><td>76.2</td><td>83.3</td><td>8130</td><td>2.45</td><td>64.5</td><td>83.7</td><td>2942</td><td>0.94</td></tr>
    <tr><td>VLM</td><td>Gemini-3.1-flash-lite</td><td>61.8</td><td>78.7</td><td>1537</td><td>0.23</td><td>42.4</td><td>51.2</td><td>1335</td><td>0.23</td></tr>
    <tr><td>VLM</td><td>Gemini-3.1-pro</td><td>83.3</td><td>90.7</td><td>5550</td><td>3.34</td><td>66.9</td><td>76.7</td><td>3959</td><td>2.49</td></tr>
    <tr><td>VLM</td><td>GPT-5-mini</td><td>67.6</td><td>77.8</td><td>2032</td><td>0.41</td><td>33.1</td><td>44.2</td><td>1437</td><td>0.30</td></tr>
    <tr><td>VLM</td><td>GPT-5</td><td>70.6</td><td>82.4</td><td>3104</td><td>3.12</td><td>37.8</td><td>51.2</td><td>2279</td><td>2.35</td></tr>
    <tr><td>VLM</td><td>GPT-5.2</td><td>70.6</td><td>83.3</td><td>1165</td><td>1.66</td><td>50.0</td><td>60.5</td><td>1022</td><td>1.54</td></tr>
    <tr><td>VLM</td><td>GPT-5.4-nano</td><td>62.0</td><td>75.9</td><td>1734</td><td>0.22</td><td>25.0</td><td>32.6</td><td>1369</td><td>0.18</td></tr>
    <tr><td>VLM</td><td>GPT-5.4-mini</td><td>63.0</td><td>77.8</td><td>2205</td><td>1.00</td><td>45.3</td><td>62.8</td><td>2224</td><td>1.05</td></tr>
    <tr><td>VLM</td><td>GPT-5.4</td><td>78.9</td><td>90.7</td><td>1299</td><td>1.98</td><td>58.1</td><td>74.4</td><td>1420</td><td>2.29</td></tr>
    <tr><td>VLM</td><td>GPT-5.5</td><td>84.0</td><td>89.8</td><td>945</td><td>2.89</td><td>58.7</td><td>74.4</td><td>1491</td><td>4.80</td></tr>
  </tbody>
</table>

</div>
</details>

<br>
<br>


🔃 **Reproducibility:** the evaluation scripts in [`scripts/`](scripts/) allow to fully reproduce our evaluation, follow the instructions below to setup the environment and get started. For size reason, the raw results of all evaluation are not provided, but per-question metrics can be found in [a `.csv` file](data/processed_results_all.csv).

---

### 🗂️ Codebase Structure
-   `conf/`: Configuration files (Hydra).
-  `notebooks/`: Jupyter notebooks for analysis and visualization.
-  **`scripts/`**: Scripts for reproducing evaluations.
-   `src/reasoning_blind_spots/`: Source code package.
    -   `dataset.py`: Dataset loading logic.
    -   `grader.py`: Scorer/Grader/Verifier logic for text outputs.
    -   `solver.py`: Solver/Generator logic for text outputs.
    -   `image_solver.py`: Custom solver for image generation tasks (OpenAI/Google).
    -   `image_grader.py`: Scorer/Grader for image generation outputs.
    -   `task.py`: Inspect AI task definition.
-   [**`main.py`**](main.py): Entry point for running the benchmark with Inspect AI.
-   [`grade_only.py`](grade_only.py): Utility script to perform grading only on provided generations.


### 🛠️ Development
If you want to contribute to the codebase, modify it or simply perform/repeat the evaluation of a model, you can follow the instructions below to set up your development environment and run the evaluation pipeline.

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

    # (Optional) if you want to use locally hosted vLLM models
    pip install -e ".[vllm]"
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
    OSS_INFERENCE_OPENAI_API=...
    ```
</details>

<details>
<summary> Running the Benchmark </summary>

To **run the benchmark** using Inspect AI you can execute the [`main.py`](./main.py) script with your chosen configuration. Default configurations are provided in [`config.yaml`](./conf/config.yaml).  Parameters can be overridden via command line arguments. Moreover, [`local_vllm.yaml`](./conf/local_vllm.yaml) provides an *untested* configuration example for local model evaluation.

Inspect AI supports various backends (OpenAI, Anthropic, Google, local models via vLLM, etc.).
For our experiments we will mainly use Gemini/OpenAI models or open-weights models. For the latter we rely on an external service hosting the models with vLLM and exposing an OpenAI API.


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

**Tool-enabled text evaluations:**
Inspect AI can now expose `code_execution()` and `web_search()` to solver models on text-output tasks with a bounded multi-step loop.

For tool-enabled runs, the relevant knobs are:
- `solver.tools.enabled`
- `solver.tools.code_execution`
- `solver.tools.web_search`
- `solver.tools.web_search_providers`
- `solver.tools.max_additional_messages`
- `sandbox` (required for client-side tool execution, e.g. `sandbox: docker`)

For `solver.tools.web_search`, native OpenAI and Gemini backends automatically use their internal web-search tool when no provider override is given. Self-hosted and OpenAI-compatible endpoints such as `openai-api/...`, AI inference services, and local vLLM require an explicit external provider, for example `solver.tools.web_search_providers=["tavily"]`. For that, a TAVILY_API_KEY will also be needed.

The model catalog in [`data/models_info.csv`](data/models_info.csv) includes a `tool_calling` column indicating whether a model can be used with text-mode tools in this benchmark, either natively or through Inspect AI tool emulation.

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
