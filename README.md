# Inference Scaling Optimization

This repository contains a Jupyter notebook that implements Monte Carlo simulations for optimizing inference scaling in AI models. It explores trade-offs between cost, time, and accuracy when using best-of-k sampling strategies across various pre-configured models.

## Features

- **Model Configurations**: Pre-defined settings for multiple AI models (e.g., GPT-5 variants, Nvidia Nemotron, Qwen3 series) including cost per token, latency, and accuracy distributions.
- **Monte Carlo Simulations**: Robust estimation of performance metrics with configurable trial counts and parallelization factors.
- **Optimization Methods**:
  - Accuracy-optimal selection
  - Cube-optimal (volume-based) selection
  - Pareto frontier analysis with utopia-closest point
  - Knee point detection using curvature or perpendicular distance
- **Interactive Visualizations**: 3D feasible space plots, accuracy vs. k curves, and cost vs. k curves with optimization markers.
- **Total Budget Constraints**: Feasibility checks based on total cost and time budgets rather than per-inference limits.

## Requirements

- Python 3.7+
- Jupyter Notebook
- Libraries: `numpy`, `matplotlib`, `ipywidgets`, `mpl_toolkits.mplot3d`

Install dependencies:
```bash
pip install numpy matplotlib ipywidgets
```

## Usage

1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook optimization_scaling.ipynb
   ```

2. Run the cells sequentially to load model configurations and functions.

3. Use the interactive widget at the end to select a model, adjust budget constraints (total cost, total time, minimum accuracy), and visualize results.

4. Key parameters:
   - `selected_model`: Choose from available models (e.g., 'gpt5', 'nvidia-nemotron-ultra-253b').
   - `C_max_total`: Maximum total cost in dollars.
   - `T_max_total`: Maximum total time in seconds.
   - `acc_min`: Minimum acceptable accuracy.
   - `k_max`: Maximum number of inferences to test.
   - `mc_trials`: Number of Monte Carlo trials for statistical robustness.
   - `parallel_factor`: Degree of parallelism (P).

## Output

- **3D Feasible Cube Plot**: Visualizes the trade-off space with constraint planes, MC trajectories, and optimal points.
- **Accuracy vs. k Plot**: Shows how accuracy improves with more inferences.
- **Total Cost vs. k Plot**: Displays cost scaling with k.
- **Text Summary**: Prints optimal k values and metrics for each method.

## Methodology

The notebook uses stochastic simulations to model variable input/output token lengths and accuracies. Optimization focuses on total budget constraints, aligning with real-world deployment scenarios where overall cost and time limits are 
fixed.

## Contributers
Thanks for collaborators

## License

MIT License