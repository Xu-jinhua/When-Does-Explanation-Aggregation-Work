# Results

The dataset-level collection contains 10 PDF reports and 54 result tables,
covering 13 distinct dataset-model combinations. Each report is distributed
with its directly compilable LaTeX source.

## Dataset reports

| Dataset | Models | PDF report | LaTeX source |
|:--|:--|:--|:--|
| ImageNet | ResNet-18, ViT-B/16 | [imagenet.pdf](pdf/by-dataset/imagenet.pdf) | [source](latex/by-dataset/imagenet/) |
| DermaMNIST | ResNet-18, ViT-B/16 | [dermamnist.pdf](pdf/by-dataset/dermamnist.pdf) | [source](latex/by-dataset/dermamnist/) |
| PathMNIST | DenseNet-121, ViT-B/16 | [pathmnist.pdf](pdf/by-dataset/pathmnist.pdf) | [source](latex/by-dataset/pathmnist/) |
| BloodMNIST | ViT-B/16 | [bloodmnist.pdf](pdf/by-dataset/bloodmnist.pdf) | [source](latex/by-dataset/bloodmnist/) |
| BreastMNIST | ViT-B/16 | [breastmnist.pdf](pdf/by-dataset/breastmnist.pdf) | [source](latex/by-dataset/breastmnist/) |
| OctMNIST | ViT-B/16 | [octmnist.pdf](pdf/by-dataset/octmnist.pdf) | [source](latex/by-dataset/octmnist/) |
| OrganCMNIST | ViT-B/16 | [organcmnist.pdf](pdf/by-dataset/organcmnist.pdf) | [source](latex/by-dataset/organcmnist/) |
| OrganSMNIST | ViT-B/16 | [organsmnist.pdf](pdf/by-dataset/organsmnist.pdf) | [source](latex/by-dataset/organsmnist/) |
| PneumoniaMNIST | ViT-B/16 | [pneumoniamnist.pdf](pdf/by-dataset/pneumoniamnist.pdf) | [source](latex/by-dataset/pneumoniamnist/) |
| RetinaMNIST | ViT-B/16 | [retinamnist.pdf](pdf/by-dataset/retinamnist.pdf) | [source](latex/by-dataset/retinamnist/) |

## Result coverage

| Dataset | Model | Protocol | Included settings |
|:--|:--|:--|:--|
| ImageNet | ResNet-18 | Paper | NAIVE, IND, NOISE-S, NOISE-K |
| ImageNet | ViT-B/16 | Paper | NAIVE, IND, NOISE-S, NOISE-K |
| DermaMNIST | ResNet-18 | Paper | NAIVE, IND, NOISE-S, NOISE-K |
| DermaMNIST | ViT-B/16 | Paper | NAIVE, IND, NOISE-S, NOISE-K |
| DermaMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |
| PathMNIST | DenseNet-121 | Generalization | NAIVE, NOISE-S, NOISE-K |
| PathMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |
| BloodMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |
| BreastMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |
| OctMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |
| OrganCMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |
| OrganSMNIST | ViT-B/16 | Full matrix | NAIVE, NOISE-S, NOISE-K |
| PneumoniaMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |
| RetinaMNIST | ViT-B/16 | Full matrix | NAIVE, IND, NOISE-S, NOISE-K |

The standalone [`naive.pdf`](pdf/naive.pdf) report contains the paper NAIVE
tables and ablations, with its source in [`latex/naive/`](latex/naive/).
