# FunnyBirds Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
> :warning: **Disclaimer**: This repository provides the minimal working code to run your own evaluations on the FunnyBirds framework. If you are looking for other components of our work, e.g., the dataset rendering or the custom evaluations, please see [here](https://github.com/visinf/funnybirds).


[R. Hesse](https://robinhesse.github.io/), [S. Schaub-Meyer](https://schaubsi.github.io/), and [S. Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp). **FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods**. _ICCV_, 2023.

## Getting Started

**Download the dataset**

```
cd /path/to/dataset/
wget ...
```

**Set up environment**



**Clone the repository**

```
git clone https://github.com/visinf/funnybirds-framework.git
cd funnybirds-framework
```

After following the above steps, you can test if everything is properly set up by running:

```
python evaluate_explainability.py --data /path/to/dataset/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
python evaluate_explainability.py --data /fastdata/rhesse/datasets/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
```

This simply evaluates the accuracy of a randomly initialized ResNet-50 and should output something like "...".


In the FunnyBirds framework each method is a combination of a model and an explanation method. 
