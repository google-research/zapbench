# Time-series forecasting

For details on the time-series forecasting models included in ZAPBench, refer to [our ICLR paper](https://openreview.net/pdf?id=oCHsDpyawq).

## Setup

```bash
# Clone the repository
git clone https://github.com/google-research/zapbench
cd zapbench

# ... potentially set up a venv or conda environment

# Install `zapbench`
pip install -e .

# Change to the time-series forecasting directory
cd zapbench/ts_forecasting
```

## Training

```bash
# See configs/ for available models and options.
python main_train.py \
  --config configs/mean.py \
  --workdir /dir/for/training
```

## Inference

```bash
# See configs/infer.py for additional options.
python main_infer.py:exp_workdir=/dir/for/training \
  --config configs/infer.py \
  --workdir /dir/for/inference
```

Metric are written to a subdirectory within `/dir/for/inference` as json files,
according to the `infer_prefix` setting in the config.

The json-files can be turned into a pandas dataframe using a utility function:

```python
from zapbench.ts_forecasting import util

df = util.get_per_step_metrics_from_directory(
  '/dir/for/inference/subdir/with/metrics',
  metric='MAE')  # or: MSE
```
