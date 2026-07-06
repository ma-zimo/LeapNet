import torch
import torch.nn as nn
import numpy as np
import random
from model import create_model
from model import create_rlmodel
from dataset import load_dataset
import argparse
from tqdm import tqdm
from torch.distributions import Bernoulli
import torchattacks
import foolbox

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("0."):
            new_k = k[2:]
        elif k.startswith("module."):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

class ModelWrapper(nn.Module):
    def __init__(self, args, basemodel, policy_model):
        super(ModelWrapper, self).__init__()
        self.base_model = basemodel
        self.policy_model = policy_model
        self.param = args

    def forward(self, inputs):
        # Use the global policy variable when calling forward
        policies = self.policy_model(inputs)
        # policies[policies < 0.5] = 0.0
        # policies[policies >= 0.5] = 1.0
        #print("policies' prob are: ", policies)
        policies = Bernoulli(policies).sample()
        # if self.param.cl_step < policies.size(1):
        #     policies[:, :-self.param.cl_step] = 1
        # policies = torch.ones_like(policies1)
        # policies = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
        return self.base_model((inputs, policies))

def evaluate(basemodel, policynet, args, val_loader, device):
    basemodel.eval()
    policynet.eval()
    model = ModelWrapper(args, basemodel, policynet)
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
            if batch_idx >= 10:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    acc = 100. * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc

def evaluate_adv(basemodel, policynet, args, val_loader, device, att, eps):

    # basemodel.eval()
    # policynet.eval()
    model = ModelWrapper(args, basemodel, policynet)
    model.eval()
    correct = 0
    total = 0
    epsilon = eps
    print(f'attack is {att}, epsilon is {epsilon}')
    # attack = torchattacks.BIM(model, eps=epsilon, alpha=2 / 255, steps=10)
    # attack = torchattacks.DeepFool(self.model, steps=step, overshoot=overshoot)
    if args.dataset == "gtsrb":
        base_model = foolbox.models.PyTorchModel(model, bounds=(-0.5, 0.5), device=device)
    elif args.dataset == "tiny_imagenet":
        base_model = foolbox.models.PyTorchModel(model, bounds=(-2.1, 2.7), device=device)
        # epsilon = epsilon * 4.8 ########foolbox need
    else:
        base_model = foolbox.models.PyTorchModel(model, bounds=(0, 1), device=device)

    if att == 'fgsm':
        attack = foolbox.attacks.FGSM()
        # attack = torchattacks.FGSM(model, eps=epsilon)
    elif att == 'pgd':
        attack = foolbox.attacks.PGD()
    elif att == 'bim':
        attack = torchattacks.BIM(model, eps=epsilon, alpha=2 / 255, steps=10)
    elif att == 'square':
        attack = torchattacks.Square(model, norm='Linf', eps=epsilon, n_queries=5000, n_restarts=1,
                                     p_init=.8, seed=0, verbose=False, loss='margin',
                                     resc_schedule=True)
    elif att == 'autoattack':
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, version='standard', n_classes=43,
                                         seed=None,
                                         verbose=False)

    for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
        if batch_idx >= 20:
            break
        images = images.to(device)
        labels = labels.to(device)
        if att == 'fgsm' or att == 'pgd':
            _, adv_images, is_adv = attack(base_model, images, labels, epsilons=epsilon)
        else:
            adv_images = attack(images, labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Total is : {total}, Test Adv Accuracy: {acc*100:.2f}%")

def get_data_info(args):
    if args.dataset == 'kul':
        return 62, 32, 32
    elif args.dataset == 'gtsrb':
        return 43, 32, 32
    elif args.dataset == 'cifar10':
        return 10, 32, 32
    elif args.dataset == 'tiny_imagenet':
        return 200, 64, 64
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture name (e.g., resnet18, efficientnet-b0)')
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['kul', 'gtsrb', 'cifar10', 'tiny_imagenet'])
    parser.add_argument('--batch_size_validation', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda" or "cpu")')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize input')
    parser.add_argument('--policy',  default=True, help='Whether to use policy model (skip_resnet)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights as base')
    parser.add_argument('--model_path', type=str, default='save_models/best_model.pth', help='Path to saved model weights')
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'base', 'cutout', 'autoaugment', 'randaugment', 'idbh'],
                        help='Augment training set.')
    parser.add_argument('--num_blocks', type=int, default=8, help='number of blocks for target model')
    parser.add_argument('--input_size', default=32, type=int, help='input images size')
    parser.add_argument('--num_classes', default=43, type=int, help='number of classes')
    parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd', 'bim', 'square', 'autoattack'])
    parser.add_argument('--eps', default=0.05, type=float, help='maximum of perturbation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    args = parser.parse_args()

    _, val_loader = load_dataset(args)
    args.num_classes, args.input_size, _ = get_data_info(args)

    SEED = 1
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

    if args.model == 'resnet18':
        args.num_blocks = 8
    elif args.model == 'resnet34':
        args.num_blocks = 3 + 4 + 6 + 3
    elif args.model == 'resnet101':
        args.num_blocks = 3 + 4 + 23 + 3
    elif args.model == 'resnet152':
        args.num_blocks = 3 + 8 + 36 + 3
    elif args.model == 'efficientnet_b0':
        args.num_blocks = 16
    elif args.model == 'mobilenet':
        args.num_blocks = 13
    else:
        print('do not know the number of the blocks')  # (int(args.model.split('-')[1]) - 4) // 6 * 3
    print('number of blocks is ', args.num_blocks)

    model = create_model(args.model, num_classes=args.num_classes, device=args.device, policy=args.policy, pretrained=args.pretrained)
    policynet = create_rlmodel(args, args.num_blocks, args.device)
    # load_model((args.model_path)
    # model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # model = model.to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    # print(checkpoint.keys())
    # checkpoint1 = torch.load('save_models/resnet18-gtsrb-LeapNet-2.pt', map_location=args.device)

    # state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    # model.load_state_dict(state_dict)
    # state_dict1 = remove_module_prefix(checkpoint['policynet_state_dict'])
    # policynet.load_state_dict(state_dict1)
    # checkpoint['policynet_state_dict'] = state_dict1
    # checkpoint['model_state_dict'] = state_dict
    # torch.save(checkpoint, args.model_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    policynet.load_state_dict(checkpoint['policynet_state_dict'])
    evaluate_adv(model, policynet, args, val_loader, args.device, args.attack, args.eps)
    print("finish")
