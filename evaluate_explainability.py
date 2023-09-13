import argparse
import random

import torch
from captum.attr import IntegratedGradients, InputXGradient

from models.resnet import resnet50
from models.vgg import vgg16
from models.model_wrapper import StandardModel
from evaluation_protocols import accuracy_protocol, controlled_synthetic_data_check_protocol, single_deletion_protocol, preservation_check_protocol, deletion_check_protocol, target_sensitivity_protocol, distractibility_protocol, background_independence_protocol
from explainers.explainer_wrapper import CaptumAttributionExplainer, CustomExplainer


parser = argparse.ArgumentParser(description='FunnyBirds - Explanation Evaluation')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset (default: imagenet)')
parser.add_argument('--model', required=True,
                    choices=['resnet50', 'vgg16'],
                    help='model architecture')
parser.add_argument('--explainer', required=True,
                    choices=['IntegratedGradients', 'InputXGradient', 'CustomExplainer'],
                    help='explainer')
parser.add_argument('--checkpoint_name', type=str, required=False, default=None,
                    help='checkpoint name (including dir)')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int,
                    help='seed')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
parser.add_argument('--nr_itrs', default=2501, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
                    
parser.add_argument('--accuracy', default=False, action='store_true',
                    help='compute accuracy')
parser.add_argument('--controlled_synthetic_data_check', default=False, action='store_true',
                    help='compute controlled synthetic data check')
parser.add_argument('--single_deletion', default=False, action='store_true',
                    help='compute single deletion')
parser.add_argument('--preservation_check', default=False, action='store_true',
                    help='compute preservation check')
parser.add_argument('--deletion_check', default=False, action='store_true',
                    help='compute deletion check')
parser.add_argument('--target_sensitivity', default=False, action='store_true',
                    help='compute target sensitivity')
parser.add_argument('--distractibility', default=False, action='store_true',
                    help='compute distractibility')
parser.add_argument('--background_independence', default=False, action='store_true',
                    help='compute background dependence')




def main():
    args = parser.parse_args()
    device = 'cuda:' + str(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create model
    if args.model == 'resnet50':
        model = resnet50(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'vgg16':
        model = vgg16(num_classes = 50)
        model = StandardModel(model)
    else:
        print('Model not implemented')
    
    if args.checkpoint_name:
        model.load_state_dict(torch.load(args.checkpoint_name, map_location=torch.device('cpu'))['state_dict'])
    model = model.to(device)
    model.eval()

    # create explainer
    if args.explainer == 'InputXGradient':
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == 'IntegratedGradients':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,256,256)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline)
    elif args.explainer == 'CustomExplainer':
        ...
    else:
        print('Explainer not implemented')

    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    if args.accuracy:
        print('Computing accuracy...')
        accuracy = accuracy_protocol(model, args)
        accuracy = round(accuracy, 5)

    if args.controlled_synthetic_data_check:
        print('Computing controlled synthetic data check...')
        csdc = controlled_synthetic_data_check_protocol(model, explainer, args)

    if args.target_sensitivity:
        print('Computing target sensitivity...')
        ts = target_sensitivity_protocol(model, explainer, args)
        ts = round(ts, 5)

    if args.single_deletion:
        print('Computing single deletion...')
        sd = single_deletion_protocol(model, explainer, args)
        sd = round(sd, 5)

    if args.preservation_check:
        print('Computing preservation check...')
        pc = preservation_check_protocol(model, explainer, args)

    if args.deletion_check:
        print('Computing deletion check...')
        dc = deletion_check_protocol(model, explainer, args)

    if args.distractibility:
        print('Computing distractibility...')
        distractibility = distractibility_protocol(model, explainer, args)

    if args.background_independence:
        print('Computing background independence...')
        background_independence = background_independence_protocol(model, args)
        background_independence = round(background_independence, 5)
    
    # select completeness and distractability thresholds such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = csdc[threshold]/3. + pc[threshold]/3. + dc[threshold]/3. + distractibility[threshold]
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    print('FINAL RESULTS:')
    print('Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS')
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(accuracy, round(csdc[best_threshold],5), round(pc[best_threshold],5), round(dc[best_threshold],5), round(distractibility[best_threshold],5), background_independence, sd, ts))
    print('Best threshold:', best_threshold)

if __name__ == '__main__':
    main()