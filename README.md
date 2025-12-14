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

To **run the benchmark** using Inspect AI using Gemini-2.5-flash-lite as the solver model, execute the following command:
```bash
inspect eval benchmark.py --model google/gemini-2.5-flash-lite
# NOTE: the verifier is hardcoded in task.py but will be configurable in the future
```

To **visualize the results**, you can use Inspect AI's built-in visualization tools:
```bash
inspect view --log-dir ./logs    
```

