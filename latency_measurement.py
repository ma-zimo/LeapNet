"""
Includes three measurement function for decision network, small target model and large target model
"""
import numpy as np
import pandas as pd
import time
import torch
import random
import seaborn as sns
import argparse
from model import create_model
from model import create_rlmodel
from torch.distributions import Bernoulli
import torch.nn as nn
colors = sns.color_palette("Blues", 100)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def measure_inference_time_policynet(policynet, input_size):
    """
                Evaluate latency of the decision network.
                """
    idx = 0
    total = 0
    policynet.eval()
    load_times = []

    inputs = torch.randn(1, 3, input_size, input_size).to(device)
    N = 30

    print("begin:")
    inference_times = []
    with torch.no_grad():
        # just warm up
        for i in range(10):
            required_latency = random.uniform(0, 3)
            latency = torch.full((inputs.size(0), 1), required_latency).to(device)
            out = policynet(inputs, latency)
        # model.reset_timings()

        for i in range(N):
            required_latency = random.uniform(0, 3)
            latency = torch.full((inputs.size(0), 1), required_latency).to(device)
            inputs = torch.randn(1, 3, input_size, input_size).to(device)
            start_time = time.time()
            out = policynet(inputs, latency)
            inference_times.append(time.time() - start_time)

    mean_value_global = np.mean(inference_times)
    min_value_global = np.min(inference_times)
    max_value_global = np.max(inference_times)
    print(f"policy inference_time_ms mean: {mean_value_global * 1000:.4f} ms")
    print(f"policy inference_time_ms min: {min_value_global * 1000:.4f} ms")
    print(f"policy inference_time_ms max: {max_value_global * 1000:.4f} ms")
    print("finished")

#### for small model with small layer dropping space, resnet18, resnet34, mobilenet
def measure_inference_time_small(model, input_size, modelname):
    """
                Evaluate performance of the model.
                """
    idx = 0
    total = 0
    model.eval()
    load_times = []

    inputs = torch.randn(1, 3, input_size, input_size).to(device)
    N = 30
    if modelname == 'resnet18':
        policy = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.]]).to(device)
        target_policy = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
    elif modelname == 'resnet34':
        policy = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).to(device)
        target_policy = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
    elif modelname == 'mobilenet':
        policy = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).to(device)
        target_policy = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]).to(device)
    data = []

    print("begin:", policy.cpu().numpy())
    inference_times = []
    with torch.no_grad():
        # just warm up
        for i in range(10):
            out = model((inputs, policy))
        # model.reset_timings()

        for i in range(N):
            inputs = torch.randn(1, 3, input_size, input_size).to(device)
            start_time = time.time()
            out = model((inputs, policy))
            inference_times.append(time.time() - start_time)

    mean_value_global = np.mean(inference_times)
    data.append({'policy': ''.join(map(str, policy.int().cpu().numpy().flatten())),
                 'mean_inference_time_ms': f"{mean_value_global * 1000:.4f}"})

    while not torch.equal(policy, target_policy):
        binary_repr = int(''.join(map(str, policy.int().cpu().numpy().flatten())), 2)
        binary_repr += 1
        if modelname == 'resnet18':
            new_policy_bits = [int(x) for x in f"{binary_repr:08b}"]
        elif modelname == 'resnet34':
            new_policy_bits = [int(x) for x in f"{binary_repr:016b}"]
        elif modelname == 'mobilenet':
            new_policy_bits = [int(x) for x in f"{binary_repr:016b}"]
        policy = torch.tensor([new_policy_bits], dtype=torch.float).cuda()
        # print("current policy:", policy.cpu().numpy())

        inference_times = []
        with torch.no_grad():
            # just warm up
            for i in range(10):
                out = model((inputs, policy))
            # model.reset_timings()

            for i in range(N):
                start_time = time.time()
                inputs = torch.randn(1, 3, input_size, input_size).to(device)
                out = model((inputs, policy))
                inference_times.append(time.time() - start_time)
            # layer_times = model.get_timings()

        mean_value_global = np.mean(inference_times)
        data.append({'policy': ''.join(map(str, policy.int().cpu().numpy().flatten())),
                     'mean_inference_time_ms': f"{mean_value_global * 1000:.4f}"})

    print("finished:", policy.cpu().numpy())
    df = pd.DataFrame(data)

    with pd.ExcelWriter(f'policy_inference_data_{modelname}_allrandom.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    print("Data has been written to Excel.")

### for large model with large layer dropping space, resnet101, resnet152
def measure_inference_time_large(model, input_size, num_blocks):
    """
                Evaluate performance of the model.
                """
    acc = 0.0
    idx = 0
    total = 0
    model.eval()
    load_times = []

    num_blocks = num_blocks
    num_samples_each_case = 200

    samples = []

    for num_ones in range(num_blocks + 1):
        base_config = [1] * num_ones + [0] * (num_blocks - num_ones)
        for _ in range(num_samples_each_case):
            shuffled_config = base_config.copy()
            random.shuffle(shuffled_config)
            samples.append(shuffled_config)

    num_random_samples = 4000
    random_samples = np.random.randint(2, size=(num_random_samples, num_blocks)).tolist()
    samples.extend(random_samples)

    samples = np.array(samples)
    print(f"Total samples: {len(samples)}")

    inputs = torch.randn(1, 3, input_size, input_size).to(device)
    N = 30
    data = []

    for sample in samples:
        policy = torch.tensor([sample], dtype=torch.float).to(device)

        inference_times = []
        with torch.no_grad():
            # just warm up
            for i in range(5):
                out = model((inputs, policy))
            # model.reset_timings()

            for i in range(N):
                start_time = time.time()
                out = model((inputs, policy))
                inference_times.append(time.time() - start_time)
            # layer_times = model.get_timings()
        mean_value_global = np.mean(inference_times)
        data.append({
            'policy': ''.join(map(str, policy.int().cpu().numpy().flatten())),
            'mean_inference_time_ms': f"{mean_value_global * 1000:.4f}"
        })

    print("finished:", policy.cpu().numpy())
    df = pd.DataFrame(data)

    with pd.ExcelWriter('policy_inference_data_resnet101.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    print("Data has been written to Excel.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture name (e.g., resnet18, efficientnet-b0)')
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['kul', 'gtsrb', 'cifar10', 'tiny_imagenet'])
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize input')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda" or "cpu")')
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

    args = parser.parse_args()


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
    policynet = create_rlmodel(args, args.num_blocks, args.device, latency=True)

    ############## load model
    # load_model((args.model_path)
    # model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    # model = model.to(args.device)
    # checkpoint = torch.load(args.model_path, map_location=args.device)
    # policynet.load_state_dict(checkpoint['policynet_state_dict'])
    # model.load_state_dict(checkpoint['model_state_dict'])

    ### measure the latency cost of the decision network
    # measure_inference_time_policynet(policynet,args.input_size)

    ### measure the latency of the target model (small), resnet18, resnet34, mobilenet
    ### we can test multiple times and collect for larget dataset that used in training predictor
    measure_inference_time_small(model, args.input_size, args.model)

    ######## measure the latency of the target model (large), resnet101, resnet152
    # measure_inference_time_large(model,args.input_size,args.num_blocks)

