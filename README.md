# FunnyBirds Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
> :warning: **Disclaimer**: This repository provides the minimal working code to run your own evaluations on the FunnyBirds framework. If you are looking for other components of our work, e.g., the dataset rendering, the custom evaluations, or the framwork code for all methods, please see [here](https://github.com/visinf/funnybirds).


[R. Hesse](https://robinhesse.github.io/), [S. Schaub-Meyer](https://schaubsi.github.io/), and [S. Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp). **FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods**. _ICCV_, 2023.

## Getting Started

The following section provides a very detailed description of how to use the FunnyBirds framework. Even if the instructions might seem a bit long and intimidating, most of the steps are finished very quickly. So don't lose hope and if you have recommendations on how to improve the framework or the instructions, we would be grateful for your feedback.

### Download the dataset

The dataset requires ~1.6GB free disk space.
```
cd /path/to/dataset/
wget download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip
unzip FunnyBirds.zip
rm FunnyBirds.zip
```

### Set up the environment

If you use conda you can create your environment as shown below:
```
conda create --name funnybirds-framework python=3.7
conda activate funnybirds-framework
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install captum -c pytorch
conda install -c conda-forge tqdm
conda install -c anaconda scipy
```

### Clone the repository

```
git clone https://github.com/visinf/funnybirds-framework.git
cd funnybirds-framework
```

After following the above steps, you can test if everything is properly set up by running:

```
python evaluate_explainability.py --data /path/to/dataset/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
python evaluate_explainability.py --data /fastdata/rhesse/datasets/funnybirds-framework/FunnyBirds --model resnet50 --explainer InputXGradient --accuracy --gpu 0
```

This simply evaluates the accuracy of a randomly initialized ResNet-50 and should output something like 
```
Acc@1   0.00 (  1.00)
Acc@5   0.00 ( 10.00)
```
followed by an error (because all metrics must be evaluate to complete the script). If this is working, we can already continue with setting up the actual evaluation. In the FunnyBirds framework each method is a combination of a model and an explanation method.

### Prepare the model

If you want to evaluate a post-hoc explanation method on the standard models ResNet-50 or VGG16, you can simple download our model weights 
```
cd /path/to/models/
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/resnet50_final_0_checkpoint_best.pth.tar
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/vgg16_final_1_checkpoint_best.pth.tar
```
and choose the models with the parameters ```--model [resnet50,vgg16] --checkpoint_name /path/to/models/model.pth.tar```. To verify this, running again
```
python evaluate_explainability.py --data /path/to/dataset/FunnyBirds --model resnet50 --checkpoint_name /path/to/models/resnet50_final_0_checkpoint_best.pth.tar --explainer InputXGradient --accuracy --gpu 0
python evaluate_explainability.py --data /fastdata/rhesse/datasets/funnybirds-framework/FunnyBirds --model resnet50 --checkpoint_name /data/rhesse/funnybirds-framework/resnet50_final_0_checkpoint_best.pth.tar --explainer InputXGradient --accuracy --gpu 0
```
should now output an accuracy score close to 1.0. If you want to use your own model, you have to **train it** and **add it to the framework**.

#### Train a new model

For training your own model please use ```train.py```.

First enter your model name to the list of valid choices of the --model argument of the parser:
```
choices=['resnet50', 'vgg16']
-->
choices=['resnet50', 'vgg16', 'your_model']
```
Next, instantiate your model, load the ImageNet weights, and change the output dimension to 50, e.g.:
```python
# create model
if args.model == 'resnet50':
    model = resnet50(pretrained=args.pretrained)
    model.fc = torch.nn.Linear(2048, 50)
elif args.model == 'vgg16':
    model = vgg16(pretrained=args.pretrained)
    model.classifier[-1] = torch.nn.Linear(4096, 50)
elif args.model == 'your_model':
    model = your_model()
    model.load_state_dict(torch.load('path/to/your/model_weights'))
    model.classifier[-1] = torch.nn.Linear(XXX, 50)
else:
    print('Model not implemented')
```

Now you can train your model by calling
```
python train.py --data /path/to/dataset/FunnyBirds --model your_model --checkpoint_dir /path/to/models/ --checkpoint_prefix your_model --gpu 0 --multi_target --pretrained --seed 0

python train.py --data /fastdata/rhesse/datasets/funnybirds-framework/FunnyBirds/ --model resnet50 --checkpoint_dir /data/rhesse/funnybirds-framework/ --checkpoint_prefix resnet50_framework --gpu 0 --multi_target --pretrained --seed 0
```
Don't forget to adjust the hyperparameters accordingly.

#### Add a new model to the framework

To add the model to the framework you first have to go to ```./models/modelwrapper.py``` and define a new class for your model that implements a ```forward()``` function and a ```load_state_dict()``` function (if ```StandardModel``` does not work for you). For examples you can refer to ```StandardModel``` or to the [complete FunnyBirds repository](https://github.com/visinf/funnybirds-framework/tree/main#train-a-new-model).
Next, you have to add the model to the choices list and the available models in ```evaluate_explainability.py``` as was done in [Train a new model](https://github.com/visinf/funnybirds-framework/tree/main#train-a-new-model). The only difference is that the weights do not need to be loaded manually.

### Prepare the explanation method

Each explanation method is wrapped in an explainer_wrapper that implements the interface functions and the function to generate the explanation:
```python
get_important_parts()
get_part_importance()
explain()
```
To implement your own wrapper, go to ```./explainers/explainer_wrapper.py``` and have a look at the ```CustomExplainer``` class. Here you can add your own explainer. If you want to evaluate an attribution method, simply let ```CustomExplainer``` inherit from ```AbstractAttributionExplainer``` and implement ```explain()``` and maybe ```__init__()```. If you want to evaluate another explanation type you also have to implement ```get_important_parts()``` and/or ```get_part_importance()```. For examples you can refer to the full [FunnyBirds repository](https://github.com/visinf/funnybirds) or the provided ```CaptumAttributionExplainer```.

The inputs and outputs of the interface functions ```get_part_importance()``` and ```get_important_parts()``` are described as comments in the code.

Finally, you have to add your CustomExplainer to the ```evaluate_explainbility.py``` script by instantiating it in:
```python
elif args.explainer == 'CustomExplainer':
    ...
```

### Run the evaluation

If you have successfully followed all of the above steps you should be able to run the evaluation using the following command:
```
python evaluate_explainability.py --data /path/to/dataset/FunnyBirds --model your_model --checkpoint_name /path/to/models/your_model_checkpoint_best.pth.tar --explainer CustomExplainer --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence --gpu 0
```
The evaluation for ResNet-50 with InputXGradient can be run with:
```
python evaluate_explainability.py --data /path/to/dataset/FunnyBirds --model resnet50 --checkpoint_name /path/to/models/resnet50_final_0_checkpoint_best.pth.tar --explainer InputXGradient --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence --gpu 0
python evaluate_explainability.py --data /fastdata/rhesse/datasets/funnybirds-framework/FunnyBirds --model resnet50 --checkpoint_name /data/rhesse/funnybirds-framework/resnet50_final_0_checkpoint_best.pth.tar --explainer InputXGradient --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence --gpu 0 
```

## Citation
If you find our work helpful please consider citing
```
@inproceedings{Hesse:2023:FunnyBirds,
  title     = {FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods},
  author    = {Hesse, Robin and Schaub-Meyer, Simone and Roth, Stefan},
  booktitle = {2023 {IEEE/CVF} International Conference on Computer Vision, {ICCV} 2021, Paris, France, October 2-6, 2023},
  year      = {2023},
  publisher = {{IEEE}}, 
  pages     = ....
}
```




