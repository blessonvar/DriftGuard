# DriftGuard

## About the Research

Federated Learning (FL) enables distributed model training across Internet of Things (IoT) devices without sharing raw data. However, in real-world deployments, data distributions on devices evolve over time—a phenomenon known as data drift. Unlike the synchronous data drift typically assumed in continual learning, drift in FL occurs *asynchronously*, with different devices shifting at different times and toward different distributions. Mitigating asynchronous data drift through FL retraining is challenging because frequent retraining incurs substantial computational overhead on resource-constrained devices, while infrequent retraining leaves drifting devices underperforming for extended periods. **DriftGuard**, a novel federated continual learning framework, is developed to address this problem.

**DriftGuard** adopts a Mixture-of-Experts (MoE) inspired architecture that decomposes the model into shared parameters capturing globally transferable knowledge and local parameters adapting to group-specific data distributions. This decomposition enables two complementary retraining strategies: *global retraining*, which updates shared parameters only when system-wide drift is detected, and *group retraining*, which selectively updates local parameters for device groups identified through a clustering mechanism based on MoE gating patterns, without requiring devices to exchange raw data. Compared to classic federated continual learning, personalized federated learning, and clustering-based methods, **DriftGuard** achieves the highest or comparable accuracy while reducing total retraining cost by up to 80%, leading to up to 2.3× higher accuracy per unit retraining cost.

## DriftGuard Framework

DriftGuard performs FL retraining under asynchronous data drift in three steps.

**Step 1**: At each time step $t$, devices perform local inference and report observations to the server.

**Step 2**: The server determines the retraining configuration $\pi^t=(Trig,S,\theta)$, which specifies whether to retrain ($Trig$), which devices participate ($S$), and which parameters to update ($\theta$). 

**Step 3**: If retraining is triggered, the selected devices perform FL retraining on the specified parameters.

<div align="center">
  <img src="figs/framework.png" width="600">

  <p><b>Fig 1. The DriftGuard Framework. </b></p>
</div>

## Code Instructions

### Quick Start

Get DriftGuard running in 5 minutes:

```bash
# 1. Setup environment
cd DriftGuard
uv sync
source .venv/bin/activate

# 2. Download datasets
gdown --fuzzy "https://drive.google.com/file/d/1L80a-vKEQnugGFvZjFYYCiqm0E8I4jVY/view?usp=sharing"
unzip datasets.zip

# 3. Start Ray first
ray start --head

# 4. Run the launcher
python src/driftguard/cli.py

# 5. View results
ls results/
```

`src/driftguard/cli.py` connects to Ray with `address="auto"`, so a Ray head node must already be running before you start the launcher.

### Environment Setup

**Requirements:**

- uv (https://docs.astral.sh/uv/getting-started/installation/)

To install the required dependencies, navigate to the `DriftGuard/` directory and run:

```bash
uv sync
source .venv/bin/activate 
```

### Dataset Preparation

In the `DriftGuard/` directory, download the dataset using:

```bash
gdown --fuzzy "https://drive.google.com/file/d/1L80a-vKEQnugGFvZjFYYCiqm0E8I4jVY/view?usp=sharing"
```

After downloading, unzip the dataset:

```bash
unzip datasets.zip
```

### Configuration

Before running the code, configure the root-level `config.json`. The current launcher reads this file directly; it does not accept `--config` or `--mode` CLI flags.

| Parameter | Description | Default Value | Type |
|-----------|-------------|----------------|------|
| `datasets` | Dataset list. Valid values in the current code are `DG5`, `PACS`, `DOMAINNET` | - | list[string] |
| `models` | Model list. Valid values are `CRST_S`, `CRST_M`, `CVIT`, `CVIT_S` | - | list[string] |
| `strategies` | Strategy list. Valid values are `Never`, `FCL-AveTrig`, `FCL-perDevice`, `PFL-AveTrig`, `PFL-perDevice`, `Cluster-based `, `DriftGuard` | - | list[string] |
| `retraining_thresholds` | Accuracy threshold used by non-DriftGuard retraining strategies | `0.01` | float |
| `global_retraining_thresholds` | Threshold used by DriftGuard global retraining logic | `0.8` | float |
| `clustering_distance` | Clustering distance threshold for grouping clients | `0.3` | float |
| `min_cluster_size` | Minimum cluster size for group retraining | `4` | int |
| `gpu_per_client` | Ray GPU fraction reserved by each client training task | `0.01` | float |
| `pi` | Enable distributed real-device scheduling with Ray custom resources | `false` | bool |

Example configuration:
```json
{
  "datasets": ["DG5"],
  "models": ["CRST_S"],
  "strategies": ["DriftGuard"],
  "retraining_thresholds": 0.01,
  "global_retraining_thresholds": 0.8,
  "clustering_distance": 0.3,
  "min_cluster_size": 4,
  "gpu_per_client": 0.01,
  "pi": false
}
```

Notes:

1. The launcher currently hardcodes several experiment-scale values inside [`src/driftguard/cli.py`](/Users/yzh/code/repos/DriftGuard/src/driftguard/cli.py), including `num_clients=2`, `total_steps=30`, `epochs=20`, and `rt_round=5`.
2. Dataset paths are resolved relative to the current working directory through `Path.cwd() / cfg.dataset.path`. In practice, run the command from the repository root.
3. For the bundled datasets, the metadata files are expected at:
   - `datasets/dg5/_meta.json`
   - `datasets/pacs/_meta.json`
   - `datasets/drift_domain_net/_meta.json`

### Running Experiments

#### Local / Single-Machine Run

On a single machine, start a local Ray head and then launch DriftGuard:

```bash
ray start --head
python src/driftguard/cli.py
```

This will:
1. Load the dataset specified in the config
2. Connect to the existing Ray runtime
3. Simulate asynchronous data drift on client devices
4. Execute the selected retraining strategies
5. Save results to the `results/` directory

#### Distributed Experiments on Real Devices

For a real distributed deployment, the current code relies on Ray cluster resources rather than `--mode server` / `--mode client` commands.

```bash
# On the server / head node
ray start --head --resources='{"server": 1}'

# On each worker device
ray start --address=<head-ip>:6379 --resources='{"client": 1, "training_node": 1}'

# Back on the server / head node
python src/driftguard/cli.py
```

Requirements for real-device runs:

1. Set `"pi": true` in `config.json`. When `pi` is `false`, the launcher does not request the custom Ray resources used for server/client placement.
2. Start Ray before launching `python src/driftguard/cli.py`.
3. Register the required custom resources on the Ray nodes. The current launcher requests:
   - `{"server": 0.01}` for the data service and federated server actors
   - `{"client": 1}` for each client actor
4. If you want training tasks to land on specific worker nodes, configure the corresponding Ray resources consistently with your cluster plan.
5. Keep the datasets on the server machine under the repository root, because the launcher resolves dataset metadata from the server-side working directory.

In other words, for a distributed real-device setup, the dataset directory must exist under the same project root where `src/` and `config.json` are located on the server.

#### Selecting Experiments

Edit `config.json` to choose the dataset, model, and strategy combinations to run. The launcher iterates over all combinations in the `datasets`, `models`, and `strategies` lists.

Examples:

```json
{
  "datasets": ["DG5"],
  "models": ["CRST_S"],
  "strategies": ["DriftGuard"]
}
```

```json
{
  "datasets": ["PACS", "DOMAINNET"],
  "models": ["CRST_M", "CVIT"],
  "strategies": ["Never", "FCL-AveTrig", "DriftGuard"]
}
```

#### Monitoring Experiments

The code will output real-time metrics and save detailed logs to the `results/` directory. You can monitor the progress by:

```bash
# View experiment results
tail -f results/[dataset]-[model]-[strategy]/log.txt
```

### Results

Experiment results are automatically saved in the `results/` directory with the following structure:

```
results/
├── [dataset]-[model]-[strategy]/
│   ├── c_0.json
│   ├── c_1.json
│   └── ... (one file per client)
└── summary.json
```

Each client result file contains:
```json
{
  "acc": [
    [timestep, accuracy],
    [timestep, accuracy],
    ...
  ],
  "cost": {
    "communication": total_communication_bytes,
    "computation": {
      "parameters": number_of_parameters_updated,
      "epochs": total_epochs_trained,
      "time_per_epoch": [epoch1_ms, epoch2_ms, ...]
    }
  }
}
```

#### Key Metrics

- **Accuracy (`acc`)**: Classification accuracy at each timestep
- **Communication Cost**: Total bytes transmitted in federated learning
- **Computation Cost**: Number of parameters updated and training time
- **Efficiency**: Accuracy per unit cost (accuracy / (communication + computation))

#### Analyzing Results

To analyze and visualize results:

```python
import json

# Load client results
with open('results/DG5-crst_s-driftguard/c_0.json', 'r') as f:
    result = json.load(f)

# Print accuracy trajectory
for timestep, acc in result['acc']:
    print(f"Timestep {timestep}: {acc:.4f}")

# Print cost breakdown
print(f"Communication: {result['cost']['communication']} bytes")
print(f"Parameters updated: {result['cost']['computation']['parameters']}")
```

## Citation

If you use DriftGuard in your research, please cite our paper:

```bibtex
@article{DriftGuard2024,
  title={DriftGuard: Adaptive Federated Learning for Model Heterogeneity and Asynchronous Data Drift},
  author={[Your Name]},
  journal={IEEE Internet of Things Journal},
  year={2024}
}
```

## Project Structure

```
DriftGuard/
├── src/driftguard/
│   ├── data/              # Data loading and preprocessing
│   ├── model/             # Model architectures (ResNet, ViT, MOE)
│   ├── federate/          # Federated learning components
│   ├── protocol/          # Communication protocols
│   ├── runtime/           # Runtime execution engine
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration management
│   ├── exp.py            # Experiment runner
│   ├── test.py           # Unit tests
│   └── recorder.py       # Results recording
├── datasets/             # Downloaded datasets
│   ├── dg5/              # DG5 dataset (Digits)
│   ├── drift_domain_net/ # Domain Net dataset
│   └── pacs/             # PACS dataset
├── results/              # Experiment results
├── figs/                 # Generated figures
├── config.json           # Configuration file
├── pyproject.toml        # Project dependencies
└── README.md             # This file
```

## Key Features

- **Asynchronous Data Drift Handling**: Detects and mitigates drift on heterogeneous clients with different time dynamics
- **Mixture-of-Experts Architecture**: Decomposes models into shared and group-specific parameters
- **Efficient Retraining**: Global and group-level retraining strategies reduce computational overhead
- **Multi-Domain Support**: Handles label shift, feature shift, and concept drift
- **Comprehensive Evaluation**: Built-in metrics for accuracy, communication cost, and computation cost
- **Flexible Deployment**: Supports both simulation and distributed testbed execution

### Retraining Strategies Comparison

| Strategy | Description | Best For | Communication Cost | Computation Cost |
|----------|-------------|----------|-------------------|-----------------|
| `DriftGuard` | Global retraining plus selective group retraining | Asynchronous drift | Low | Low |
| `FCL-AveTrig` | Retrain all clients when average accuracy drops below threshold | Uniform degradation | Medium | Medium |
| `FCL-perDevice` | Retrain only clients with degraded accuracy | Sparse client drift | Medium | Medium |
| `PFL-AveTrig` | MoE-style personalized retraining triggered by average accuracy | Personalized adaptation | Medium | High |
| `PFL-perDevice` | MoE-style personalized retraining triggered per client | Non-IID and sparse drift | Medium | High |
| `Cluster-based ` | Cluster selected clients and retrain by group | Group-structured drift | Low-Medium | Medium |
| `Never` | Disable retraining and only track inference accuracy | Ablation / baseline | Very Low | Very Low |

## Performance Highlights

DriftGuard achieves:
- Up to **80% reduction in retraining cost** compared to classic federated continual learning
- Up to **2.3× higher accuracy per unit retraining cost** compared to baseline methods
- Comparable or superior accuracy across multiple datasets (DG5, Drift-Domain-Net, PACS)

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```bash
# Solution: reduce the workload in src/driftguard/cli.py
# The current launcher hardcodes batch_size, num_clients, epochs, and total_steps there.
```

**Issue**: Dataset files not found
```bash
# Solution: Ensure datasets are properly extracted
cd /Users/yzh/code/repos/DriftGuard
unzip datasets.zip
```

**Issue**: Slow training on CPU
```bash
# Solution: the code is GPU-oriented. For CPU-only execution, reduce the workload in src/driftguard/cli.py
```

### Getting Help

For issues and questions:
1. Check the [Issues](https://github.com/yourusername/DriftGuard/issues) page
2. Review existing documentation in `src/driftguard/` module docstrings
3. Check logs in `results/[experiment]/log.txt`

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This work builds upon federated learning, continual learning, and domain adaptation research. We thank the authors of the foundational datasets used in this work.
