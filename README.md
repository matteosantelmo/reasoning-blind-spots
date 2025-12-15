# reasoning-blind-spots

Reasoning Blind Spots is a benchmark designed to stress test the reasoning capabilities of frontier AI models on tasks that are straightforward for humans. The questions in the benchmark are crafted to highlight the limitations of current AI systems and understand where they struggle in reasoning tasks.
This benchmark was developed as part of the _"Reasoning in AI"_ course at EPFL (MATH-700).
Our benchmark relies on [Inspect AI](https://inspect.aisi.org.uk) as the evaluation framework.

## Environment Setup
To set up the environment for this project, follow these steps:

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .[dev]
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

### Repository Structure

-   `conf/`: Configuration files (Hydra).
-   `data/`: Contains the dataset files.
-   `src/reasoning_blind_spots/`: Source code package.
    -   `dataset.py`: Dataset loading logic.
    -   `scorer.py`: Verifier/Scorer logic.
    -   `task.py`: Inspect AI task definition.
-   `benchmark.py`: Entry point for running the benchmark with Inspect AI.
-   `pyproject.toml`: Project metadata and dependencies.

## Running the Benchmark

To **run the benchmark** using Inspect AI you can execute the [`main.py`](./main.py) script with your chosen configuration. Default configurations are provided in [`config.yaml`](./conf/config.yaml), but parameters can be overridden via command line arguments.

Example command to run the benchmark with a specific model:
```bash
python main.py \
    solver.model_name="gemini-2.0-flash-lite" \
    solver.backend="google" \
    solver.generate_config.reasoning_tokens=0
```

To **visualize the results**, you can use Inspect AI's built-in visualization tools:
```bash
inspect view --log-dir ./outputs/<your-log-dir>/
```
