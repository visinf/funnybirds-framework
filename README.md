# FunnyBirds Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
> :warning: **Disclaimer**: This repository provides the minimal working code to run your own evaluations on the FunnyBirds framework. If you are looking for other components of our work, e.g., the dataset rendering, the custom evaluations, or the framwork code for all methods, please see [here](https://github.com/visinf/funnybirds).


[R. Hesse](https://robinhesse.github.io/), [S. Schaub-Meyer](https://schaubsi.github.io/), and [S. Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp). **FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods**. _ICCV_, 2023.

## Getting Started

### Download the dataset

```
cd /path/to/dataset/
wget ...
```

### Set up the environment



### Clone the repository

```
git clone https://github.com/visinf/funnybirds-framework.git
cd funnybirds-framework
```

After following the above steps, you can test if everything is properly set up by running:

```
python evaluate_explainability.py --data /path/to/dataset/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
python evaluate_explainability.py --data /fastdata/rhesse/datasets/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
```

This simply evaluates the accuracy of a randomly initialized ResNet-50 and should output something like "...". If this is working, we can already continue with setting up the actual evaluation. In the FunnyBirds framework each method is a combination of a model and an explanation method.

### Prepare the model

If you want to evaluate a post-hoc explanation method on the standard models ResNet-50 or VGG16, you can simple download our model weights 
```
cd /path/to/models/
wget ...
```
and chooses the models with the parameters ```--model [resnet50,vgg16] --checkpoint_name /path/to/models/model.pth.tar```. To verify this, running again
```
python evaluate_explainability.py --data /path/to/dataset/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
python evaluate_explainability.py --data /fastdata/rhesse/datasets/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
```
should now output an accuracy score close to 1.0. If you want to use you own model, you have to **train it** and **add it to the framework**.

#### Train a new model
....

#### Add a new model to the framework

### Prepare the explanation method

Each explanation method is wrapped in an explainer_wrapper that implements the interface functions and the function to generate the explanation:
```python
get_important_parts()
get_part_importance()
explain()
```
To implement your own wrapper go to ```./explainers/explainer_wrapper.py``` and have a look at the ```CustomExplainer``` class.

## Citation
If you find our work helpful please consider citing
```
@inproceedings{Hesse:2023:FunnyBirds,
  title     = {FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods},
  author    = {Hesse, Robin and Schaub-Meyer, Simone and Roth, Stefan},
  booktitle = {2023 {IEEE/CVF} International Conference on Computer Vision, {ICCV} 2021, Paris, France, October 2-6, 2023},
  year      = {2023},
  publisher    = {{IEEE}}, 
  pages     = ....
}
```




