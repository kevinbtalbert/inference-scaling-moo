# Inference Scaling - Multidimensional Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2501.18905-b31b1b.svg)](https://arxiv.org/abs/2510.18905)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-success.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourname/inference-scaling.svg?style=social&label=Star)](https://github.com/yourname/inference-scaling)

Efficient AI inference scaling is essential for practical deployment. Instead of relying on static scaling heuristics or simple bivariate trade-offs between performance and compute, we should incorporate multiple factors such as cost, latency, and accuracy. This project models inference scaling as a multi-objective optimization (MOO) problem and simulates it using 3D and 2D space. Below is the sample output:

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8ee334c9-7aaa-45a7-96c4-d2f7b65e6802" width="380" /></td>
    <td><img src="https://github.com/user-attachments/assets/3818d179-ab64-4826-824d-91e0072f42f3" width="380" /></td>
    <td><img src="https://github.com/user-attachments/assets/43e62b53-ed3c-4040-a1e6-3cab029e2791" width="380" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ead77660-e598-44ff-ad0d-ce84831a4f2c" width="380" /></td>
    <td><img src="https://github.com/user-attachments/assets/19464d87-8fe9-4f9f-9a6c-183f8de6c27b" width="380" /></td>
    <td><img src="https://github.com/user-attachments/assets/54ca3352-e262-479d-89ea-1e57db0b59a1" width="380" /></td>
  </tr>
</table>


A feasible space shaped by 3D constraints captures values that a 2D space fails to account for.
<img width="512" height="256" alt="image" src="https://github.com/user-attachments/assets/86edd5cf-b41c-44e3-82f9-8a6bc3d74c08" />

This repository contains a Jupyter notebook that implements Monte Carlo simulations for optimizing inference scaling in AI models. It explores trade-offs between cost, time, and accuracy across various pre-configured models.
## Project Structure
```
inf_opt/
├── 01_Installer/
│   ├── install.sh
│   └── requirements.txt
├── 02_MultiobjectiveOptimization/
│   ├── __init__.py
│   ├── Inference_scaling_MOO.ipynb
│   ├── inference_scaling.py
│   └── __pycache__/
├── .project-metadata.yaml
├── pyproject.toml
└── README.md
```


## Features

- **Model Configurations**: Pre-defined settings for multiple AI models (e.g., GPT-5 variants, Nvidia Nemotron, Qwen3 series), including cost per token, latency, and accuracy distributions.
- **Monte Carlo Simulations**: Statistical estimation of performance metrics with configurable trial counts and parallelization factors.
- **Optimization Methods**:
  - 1. Maximal Accuracy selection
  - 2. Maximal Cube selection
  - 3. Pareto frontiers and utopia-closest point
  - 4. Pareto frontiers and Knee point detection
- **Interactive Visualizations**: 3D feasible space plots
- **Constraints**: Feasibility checks based on total cost and time budgets, and minimal accuracy requirements.


## Requirements
- Python 3.7+
- Jupyter Notebook
- Libraries: `numpy`, `matplotlib`, `ipywidgets`, `mpl_toolkits.mplot3d`, `pyyaml`

## Installation

Run the provided shell script to install dependencies automatically:

```bash
./01_Installer/install.sh
```

Or install manually:
```bash
pip install -r 01_Installer/requirements.txt
```

## Repository Organization

- `01_Installer/`: Contains installation scripts (`install.sh`), dependency list (`requirements.txt`), and packaging configuration (`pyproject.toml`).
- `02_MultiobjectiveOptimization/`: Core Python module (`inference_scaling.py`) and the interactive Jupyter notebook (`Inference_scaling_MOO.ipynb`).

## Usage

1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook 02_MultiobjectiveOptimization/Inference_scaling_MOO.ipynb
   ```

2. Run the cells sequentially to load model configurations and functions.

3. Use the interactive widget at the end to select a model, adjust budget constraints (max-cost, max-time, min-accuracy), and visualize results.

4. Key parameters:
   - `selected_model`: Choose from available sample models (e.g., 'gpt5', 'nvidia-nemotron-ultra-253b').
   - `C_max_total`: Maximum total cost in dollars.
   - `T_max_total`: Maximum total time in seconds.
   - `acc_min`: Minimum acceptable accuracy.
   - `k_max`: Maximum number of inferences to test.
   - `mc_trials`: Number of Monte Carlo trials for statistical robustness.
   - `parallel_factor`: Degree of parallelism (P).

## Output

- **3D Feasible Cube Plot**: Visualizes the trade-off in the 3D space with constraint planes, MC trajectories, and optimal points (optimality could be different for priority).
- **Accuracy vs. k Plot**: Shows how accuracy improves with more inferences.
- **Total Cost vs. k Plot**: Displays cost scaling with k.
- **Text Summary**: Prints optimal k values and metrics for each method.

## Methodology
The notebook uses stochastic simulations to model variable input/output token lengths and accuracies. Optimization focuses on total budget constraints, aligning with real-world deployment scenarios where overall cost and time limits are 
fixed.

## Contributers
Thanks to collaborators

## License
MIT License
