# ZAPBench ⚡

The Zebrafish Activity Prediction Benchmark (ZAPBench) measures progress on the
 problem of predicting cellular-resolution neural activity throughout an entire
 vertebrate brain. For more information, refer to [our ICLR paper](https://openreview.net/pdf?id=oCHsDpyawq) and the [companion website](https://google-research.github.io/zapbench).

## Getting started

To get started with ZAPBench, we provide tutorial-style notebooks in the `colabs/` directory:

- **Datasets:** Overview of various datasets we released and how to access them. [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-reseach/zapbench/blob/main/colabs/datasets.ipynb)
- **Training and evaluation:** How to train and evaluate forecasting methods on ZAPBench in a framework agnostic way. [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-reseach/zapbench/blob/main/colabs/train_and_eval.ipynb)
- **Metrics:** Explains how to load predictions made by the methods reported in the paper for additional analyses, e.g., to compute custom metrics. [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-reseach/zapbench/blob/main/colabs/metrics.ipynb)
- **Interactive time-series forecasting:** Shows how to run a `jax` time-series forecasting model interactively. [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-reseach/zapbench/blob/main/colabs/ts_forecasting_interactive.ipynb)

## Contents

In addition, this repository contains:

- Code for the forecasting models used in the paper, implemented in `jax`, in the `zapbench/models/` subdirectory.
- Scripts and configs to train and evaluate time-series and video forecasting models, in `zapbench/ts_forecasting/` and `zapbench/video_forecasting/`, respectively. The READMEs in those subdirectories contain further usage instructions.
- Config for alignment and normalization pipeline of the raw data in `processing/alignment_and_normalization.gin`; see file header for usage.
- Notebook demonstrating how to load the FFN checkpoint used for segmentation in `processing/ffn_inference.ipynb`.
- Notebook loading and plotting raw stimulus time-series in `processing/stimuli.ipynb`.
- A WebGL-viewer for calcium fluorescence data in `fluroglancer/`.

## Datasets

[Further information on associated datasets](http://zapbench-release.storage.googleapis.com/volumes/README.html).


## License

Apache 2.0

*This is not an officially supported Google product.*
