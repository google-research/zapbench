# Video forecasting

For details on the video forecasting models included in ZAPBench, refer to [our ICLR paper](https://openreview.net/pdf?id=oCHsDpyawq).

## Setup

```bash
# Clone the repository
git clone https://github.com/google-research/zapbench
cd zapbench

# ... potentially set up a venv or conda environment

# Install `zapbench`
pip install -e .

# Change to the video forecasting directory
cd zapbench/video_forecasting
```

## Training

```bash
# See configs/ for available models and options.
python main_train.py \
  --config configs/unet_test.py \
  --workdir /dir/for/training
```

## Inference

```bash
# See configs/infer.py for additional options.
python main_infer.py:/dir/for/inference \
  --config configs/infer.py \
  --workdir /dir/for/training
```

Metric are written to a subdirectory within `/dir/for/inference` as json files,
according to the `json_path_prefix` setting in the config.

The json-files can be turned into a pandas dataframe using a utility function:

```python
from zapbench.ts_forecasting import util

df = util.get_per_step_metrics_from_directory(
  '/dir/for/inference/subdir/with/metrics',
  metric='trace_step_mae')  # or: trace_step_mse
```
