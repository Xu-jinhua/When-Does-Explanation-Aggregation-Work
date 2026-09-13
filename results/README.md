# Results

The dataset-level collection contains 14 PDF reports with 711 non-empty result
tables. Each report is distributed with its directly compilable LaTeX source;
the complete inventory is recorded in `latex/RESULT_MANIFEST.json`.

## Dataset reports

| Dataset | Models | PDF report | LaTeX source |
|:--|:--|:--|:--|
| ImageNet | DenseNet-121, ResNet-18, ViT-B/16 | [imagenet.pdf](pdf/by-dataset/imagenet.pdf) | [source](latex/by-dataset/imagenet/) |
| DermaMNIST | ResNet-18, ViT-B/16 | [dermamnist.pdf](pdf/by-dataset/dermamnist.pdf) | [source](latex/by-dataset/dermamnist/) |
| PathMNIST | DenseNet-121, ViT-B/16 | [pathmnist.pdf](pdf/by-dataset/pathmnist.pdf) | [source](latex/by-dataset/pathmnist/) |
| BloodMNIST | ViT-B/16 | [bloodmnist.pdf](pdf/by-dataset/bloodmnist.pdf) | [source](latex/by-dataset/bloodmnist/) |
| BreastMNIST | ViT-B/16 | [breastmnist.pdf](pdf/by-dataset/breastmnist.pdf) | [source](latex/by-dataset/breastmnist/) |
| OCTMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [octmnist.pdf](pdf/by-dataset/octmnist.pdf) | [source](latex/by-dataset/octmnist/) |
| OrganCMNIST | ViT-B/16 | [organcmnist.pdf](pdf/by-dataset/organcmnist.pdf) | [source](latex/by-dataset/organcmnist/) |
| OrganSMNIST | ViT-B/16 | [organsmnist.pdf](pdf/by-dataset/organsmnist.pdf) | [source](latex/by-dataset/organsmnist/) |
| PneumoniaMNIST | ViT-B/16 | [pneumoniamnist.pdf](pdf/by-dataset/pneumoniamnist.pdf) | [source](latex/by-dataset/pneumoniamnist/) |
| RetinaMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [retinamnist.pdf](pdf/by-dataset/retinamnist.pdf) | [source](latex/by-dataset/retinamnist/) |
| Food101 | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18 | [food101.pdf](pdf/by-dataset/food101.pdf) | [source](latex/by-dataset/food101/) |
| OrganAMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [organamnist.pdf](pdf/by-dataset/organamnist.pdf) | [source](latex/by-dataset/organamnist/) |
| Places365 | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [places365.pdf](pdf/by-dataset/places365.pdf) | [source](latex/by-dataset/places365/) |
| TissueMNIST | DenseNet-121, EfficientNet-B0, MobileNetV3-Large, ResNet-18, ResNet-50, ViT-B/16 | [tissuemnist.pdf](pdf/by-dataset/tissuemnist.pdf) | [source](latex/by-dataset/tissuemnist/) |

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
