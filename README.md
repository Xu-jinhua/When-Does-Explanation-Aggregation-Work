# When Does Aggregating Explanations Work?

Jinhua Xu, Davide Anguita, Fabio Roli, Jing Yuan, and Luca Oneto

## Abstract

Post-hoc explanation methods for image classifiers often produce inconsistent
results, particularly under data perturbations, model variations, or when
different explanation techniques are applied. Ensembling multiple explanations
has therefore emerged as a common strategy to mitigate such disagreement.
However, existing approaches remain largely empirical and offer limited
theoretical insight into the conditions under which ensemble methods improve
reliability or, conversely, fail. In this paper, we study the aggregation of
image explanations derived from patch-based saliency maps. We first rank image
patches according to their estimated importance and then combine these rankings
using different rank-aggregation strategies. Through experiments on multiple
datasets, model architectures, post-hoc explanation methods, and
rank-aggregation strategies, we show that the effectiveness of aggregation
methods depends on how well two assumptions are met: independence among the
aggregated explanations and compatibility of the explanation noise with a
distance-based rank-noise model.

## Results

The experimental results are organized as dataset-level reports. Each report
collects the released model and aggregation-setting results for one dataset;
the compiled PDF and its LaTeX source are published together.

| Dataset | PDF report | LaTeX source |
|:--|:--|:--|
| ImageNet | [PDF](results/pdf/by-dataset/imagenet.pdf) | [source](results/latex/by-dataset/imagenet/) |
| DermaMNIST | [PDF](results/pdf/by-dataset/dermamnist.pdf) | [source](results/latex/by-dataset/dermamnist/) |
| PathMNIST | [PDF](results/pdf/by-dataset/pathmnist.pdf) | [source](results/latex/by-dataset/pathmnist/) |
| BloodMNIST | [PDF](results/pdf/by-dataset/bloodmnist.pdf) | [source](results/latex/by-dataset/bloodmnist/) |
| BreastMNIST | [PDF](results/pdf/by-dataset/breastmnist.pdf) | [source](results/latex/by-dataset/breastmnist/) |
| Food101 | [PDF](results/pdf/by-dataset/food101.pdf) | [source](results/latex/by-dataset/food101/) |
| OCTMNIST | [PDF](results/pdf/by-dataset/octmnist.pdf) | [source](results/latex/by-dataset/octmnist/) |
| OrganAMNIST | [PDF](results/pdf/by-dataset/organamnist.pdf) | [source](results/latex/by-dataset/organamnist/) |
| OrganCMNIST | [PDF](results/pdf/by-dataset/organcmnist.pdf) | [source](results/latex/by-dataset/organcmnist/) |
| OrganSMNIST | [PDF](results/pdf/by-dataset/organsmnist.pdf) | [source](results/latex/by-dataset/organsmnist/) |
| Places365 | [PDF](results/pdf/by-dataset/places365.pdf) | [source](results/latex/by-dataset/places365/) |
| PneumoniaMNIST | [PDF](results/pdf/by-dataset/pneumoniamnist.pdf) | [source](results/latex/by-dataset/pneumoniamnist/) |
| RetinaMNIST | [PDF](results/pdf/by-dataset/retinamnist.pdf) | [source](results/latex/by-dataset/retinamnist/) |
| TissueMNIST | [PDF](results/pdf/by-dataset/tissuemnist.pdf) | [source](results/latex/by-dataset/tissuemnist/) |

## Code

The implementation is available in [`code/`](code/), with configuration files
in [`code/configs/`](code/configs/) and the command-line entry point `xai-exp`.
The code is organized around the following stages:

### Datasets

The experiments use ImageNet, MedMNIST, Food-101, and Places365. The official
dataset pages are provided below for data access and download.

| Dataset | Official page |
|:--|:--|
| ImageNet | [image-net.org](https://www.image-net.org/) |
| MedMNIST | [medmnist.com](https://medmnist.com/) |
| Food-101 | [ETH Zurich dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) |
| Places365 | [MIT Places365](http://places2.csail.mit.edu/) |

Dataset manifests and preprocessing settings are defined in
[`code/src/xai_ensemble/data/`](code/src/xai_ensemble/data/) and
[`code/configs/`](code/configs/).

### Models

The collection uses the following image classification architectures:

| Model | Source |
|:--|:--|
| ResNet-18 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| ResNet-50 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| DenseNet-121 | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) |
| EfficientNet-B0 | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) |
| MobileNetV3-Large | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) |
| ViT-B/16 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) |

### Data Preparation and Model Training (Phase 0)

Phase 0 prepares the data manifests, class stratified partitions, reference
models, checkpoints, and train split statistics used by the later stages.

Source: [`code/src/xai_ensemble/phase0/`](code/src/xai_ensemble/phase0/)

### Attribution Generation Process (Phase 1)

Phase 1 generates post hoc attribution maps for each configured model, dataset,
perturbation condition, and explanation method. The resulting artifacts are
immutable and include their provenance metadata.

Source: [`code/src/xai_ensemble/phase1/`](code/src/xai_ensemble/phase1/)

### Explanation Aggregation and Evaluation (Phase 2)

Phase 2 ranks image patches, aggregates the explanation rankings with the
configured ensemble rules, and evaluates fidelity, consistency, and robustness
through the masking game.

Source: [`code/src/xai_ensemble/phase2/`](code/src/xai_ensemble/phase2/)
