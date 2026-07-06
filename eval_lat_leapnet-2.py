import torch
import torch.nn as nn
import numpy as np
import random
from model import create_model
from model import create_rlmodel
from model import create_latpredictor
from dataset import load_dataset
import argparse
from tqdm import tqdm
from torch.distributions import Bernoulli
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class latencyWrapper(nn.Module):
    def __init__(self, args, model, policy_model):
        super(latencyWrapper, self).__init__()
        base_model = model
        policy_model = policy_model
        param = args

    def forward(self, inputs):
        # Use the global policy variable when calling forward
        # latency = torch.full((inputs.size(0), 1), 2.000).to(device)
        latency = torch.FloatTensor(inputs.size(0), 1).uniform_(0.5, 4.5).to(self.param.device)
        policies = self.policy_model(inputs, latency)
        #print("policies' prob are: ", policies)
        policies = Bernoulli(policies).sample()
        return self.base_model((inputs, policies))


def evaluate(basemodel, policynet, args, val_loader, device):
    basemodel.eval()
    policynet.eval()
    model = latencyWrapper(args, basemodel, policynet)
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

def eval_latency_frames(basemodel, policynet, latency_predictor, args, device):

    inference_time = 0
    idx = 0
    total = 0
    basemodel.eval()
    policynet.eval()
    latency_predictor.eval()
    model.eval()
    measured_latency = []
    inference_times = []
    given_latency = [1.4,1.1,1.8] # you can change from 0.5-2 in our default setting (resnet18 on RTX2080)
    given_frames = [80,120,100]  # you can change arbitrarily, but the current code only support three-stage processing
    input_latency = []
    predicted_latency = []
    latency_loss = 0
    unique_policies = set()
    unique_probs = set()

    with torch.no_grad():
        ############# just warm up
        inputs = torch.randn(1, 3, 32, 32).to(device)
        print("inputs type:", inputs.dtype)
        for i in range(50):
            inputs = torch.randn(1, 3, 32, 32).to(device)

            # policy = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
            latency = torch.full((1, 1), given_latency[0]).to(device)
            probs = policynet(inputs, latency)
            ####### Sample method
            policy = Bernoulli(probs).sample()
            predicted_l = latency_predictor(probs)
            # print('predicted latency based on probs is', predicted_l)
            out = basemodel((inputs, policy))
        # basemodel.reset_timings()

        for i in range(given_frames[0]):
            idx += 1
            inputs = torch.randn(1, 3, 32, 32).to(device)
            inputs = torch.clamp(inputs, min=-0.5, max=0.5)
            # policy = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
            latency = torch.full((1, 1), given_latency[0]).to(device)
            latency_value = latency.sum().item()
            input_latency.append(latency_value)
            probs = policynet(inputs, latency)
            ####### Sample method
            policy = Bernoulli(probs).sample()
            predicted_l = latency_predictor(probs)
            # print('predicted latency based on probs is', predicted_l)
            start_time = time.time()
            out = basemodel((inputs, policy))
            inference_time = time.time() - start_time
            measured_latency.append(inference_time * 1000)
            predicted_l = latency_predictor(policy)  ########## probs for plot1; policy for plot2
            predicted_latency.append(predicted_l.item())

        for i in range(given_frames[1]):
            idx += 1
            inputs = torch.randn(1, 3, 32, 32).to(device)
            inputs = torch.clamp(inputs, min=-0.5, max=0.5)
            # policy = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
            latency = torch.full((1, 1), given_latency[1]).to(device)
            latency_value = latency.sum().item()
            input_latency.append(latency_value)
            probs = policynet(inputs, latency)
            ####### Sample method
            policy = Bernoulli(probs).sample()
            predicted_l = latency_predictor(probs)
            # print('predicted latency based on probs is', predicted_l)
            start_time = time.time()
            out = basemodel((inputs, policy))
            inference_time = time.time() - start_time
            measured_latency.append(inference_time * 1000)
            predicted_l = latency_predictor(policy)  ########## probs for plot1; policy for plot2
            predicted_latency.append(predicted_l.item())

        for i in range(given_frames[2]):
            idx += 1
            inputs = torch.randn(1, 3, 32, 32).to(device)
            inputs = torch.clamp(inputs, min=-0.5, max=0.5)
            # policy = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
            latency = torch.full((1, 1), given_latency[2]).to(device)
            latency_value = latency.sum().item()
            input_latency.append(latency_value)
            probs = policynet(inputs, latency)
            ####### Sample method
            policy = Bernoulli(probs).sample()
            predicted_l = latency_predictor(probs)
            # print('predicted latency based on probs is', predicted_l)
            start_time = time.time()
            out = basemodel((inputs, policy))
            inference_time = time.time() - start_time
            measured_latency.append(inference_time * 1000)
            predicted_l = latency_predictor(policy)  ########## probs for plot1; policy for plot2
            predicted_latency.append(predicted_l.item())

    # latency_loss = abs(1 - predicted_latency / input_latency).sum()
    average_latency = np.mean(predicted_latency)
    print("averaged predicted_latency:", average_latency)
    print('lens of given {}, lens of pred {}'.format(len(input_latency), len(predicted_latency)))
    # print('measured_latency',measured_latency)
    num_frames = sum(given_frames)  # total frames

    # compute the avarge latency for each part
    avg_predicted_latency = []
    start = 0
    for segment_size in given_frames:
        segment = predicted_latency[start:start + segment_size]
        avg_predicted_latency.extend([sum(segment[:j + 1]) / len(segment[:j + 1]) for j in range(len(segment))])
        start += segment_size
    avg_measured_latency = []
    start = 0
    for segment_size in given_frames:
        segment = measured_latency[start:start + segment_size]
        avg_measured_latency.extend([sum(segment[:j + 1]) / len(segment[:j + 1]) for j in range(len(segment))])
        start += segment_size


    data = {
        "Frame Index": list(range(num_frames)),
        "Predicted Latency": predicted_latency,
        "Measured Latency": measured_latency,
        "Target Latency": input_latency,
        "Avg. Predicted Latency": avg_predicted_latency,
        "Avg. Measured Latency": avg_measured_latency
    }

    df = pd.DataFrame(data)
    df.to_excel(f'latency_data.xlsx', index=False)  # {directory}/test/
    print(f"data has saved to latency_data.xlsx")

    return
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

# 提取数据
def extract_data(df):
    frame_index = df['Frame Index']
    predicted_latency = df['Predicted Latency']
    measured_latency = df['Measured Latency']
    input_latency = df['Target Latency']
    avg_predicted_latency = df['Avg. Predicted Latency']
    avg_measured_latency = df['Avg. Measured Latency']
    return frame_index, predicted_latency, measured_latency, input_latency, avg_predicted_latency, avg_measured_latency

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture name (e.g., resnet18, efficientnet-b0)')
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['kul', 'gtsrb', 'cifar10', 'tiny_imagenet'])
    parser.add_argument('--batch_size_validation', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use (e.g., "cuda" or "cpu")')
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
    parser.add_argument('--eps', default=0.06, type=float, help='input images channel')
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

    model = create_model(args.model, num_classes=args.num_classes, device=args.device, policy=args.policy, pretrained=args.pretrained, inference=True)
    policynet = create_rlmodel(args, args.num_blocks, args.device, latency = True)
    # load_model((args.model_path)
    # model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # model = model.to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    policynet.load_state_dict(checkpoint['policynet_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    latency_predictor = create_latpredictor(args.device, args.num_blocks)
    latency_predictor.load_state_dict(torch.load(f'./latency_predictor_{args.model}.pth.tar'))
    # evaluate(model, policynet, args, val_loader, args.device)
    eval_latency_frames(model, policynet, latency_predictor, args, args.device)

    ############ plot figures
    ## load data
    df_resnet18 = pd.read_excel('latency_data.xlsx')

    frame_index_18, predicted_latency_18, measured_latency_18, input_latency_18, avg_predicted_latency_18, avg_measured_latency_18 = extract_data(
        df_resnet18)

    deep_purple_color = '#4B0082'
    blue_color = '#1f77b4'  # A nice shade of blue
    orange_color = '#ff7f0e'  # A vibrant orange
    green_color = '#2ca02c'  # A pleasing green
    colors = sns.color_palette("Blues", 100)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))

    axes[0].plot(frame_index_18, predicted_latency_18, "--", linewidth=4, color=deep_purple_color,
                 label="Predicted Latency")
    axes[0].plot(frame_index_18, measured_latency_18, "-", linewidth=3, color=green_color, label="Measured Latency")
    axes[0].plot(frame_index_18, input_latency_18, ":", linewidth=2, color="red", label="Target Latency")
    # axes[0].set_title('ResNet18')
    axes[0].grid(False)
    axes[0].tick_params(axis='both', labelsize=22)

    axes[1].plot(frame_index_18, avg_predicted_latency_18, "--", linewidth=4, color=deep_purple_color,
                 label="Predicted Latency")
    axes[1].plot(frame_index_18, avg_measured_latency_18, "-", linewidth=3, color=green_color,
                 label="Measured Latency")
    axes[1].plot(frame_index_18, input_latency_18, ":", linewidth=2, color="red", label="Target Latency")
    axes[1].grid(False)
    axes[1].tick_params(axis='both', labelsize=22)
    axes[1].set_xlabel("Frame Index", fontsize=26)
    titles = ['(b) ResNet18 on RTX 2080Ti']

    # set x range
    for ax in axes.flat:
        ax.set_xlim([-0.5, 300.5])

    #  legend
    handles, labels = axes[0].get_legend_handles_labels()
    new_order = [2, 0, 1]  # original order is Predicted Latency, Measured Latency, Target Latency
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]
    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        fontsize=16,
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
        handletextpad=0.3,
        columnspacing=0.7
    )
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor("black")

    for ax in axes.flat:
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)

    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    # save figure
    fig.savefig('latency_result.pdf', dpi=300, bbox_inches="tight")
    fig.savefig('latency_result.jpg', dpi=300, bbox_inches="tight")
    # plt.show()


