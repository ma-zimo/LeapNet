import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model
from model import create_rlmodel
from dataset import load_dataset
import argparse
import os
from tqdm import tqdm
from torch.distributions import Bernoulli

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

def train(model, args, train_loader, val_loader, device, epochs=50):
    # optimizer = optim.SGD([
    #     {"params": model.features.parameters(), "lr": 1e-4},
    #     {"params": model.classifier.parameters(), "lr": 1e-3},
    # ], momentum=0.9, weight_decay=5e-4)

    optimizer = optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4
    )

    # 学习率调度器：每 20 epoch 衰减一次学习率，衰减因子为 0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            policy = torch.ones(images.size(0), args.num_blocks, dtype=torch.int64).to(device)

            optimizer.zero_grad()
            outputs = model((images, policy))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        val_acc = validate_pretrain(model, args, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'save_models/{args.model}_{args.dataset}_best_pretrained.pth')

def get_reward(args, preds, targets, policy):

    block_use = policy.sum(1).float() / policy.size(1)
    sparse_reward = 1.0 - block_use ** 2

    _, pred_idx = preds.max(1)
    match = (pred_idx == targets)
    match_int = match.int()
    # print('match is {}, pred_idx is {}, targets is {}'.format(match_int.float(),pred_idx, targets))

    reward = sparse_reward.clone()  # Clone sparse_reward to avoid modifying it in-place
    reward[~match] = args.penalty
    reward = reward.unsqueeze(1)
    # print('reward is',reward)

    return reward, match_int.float()

def train_policy(base_model, policynet, args, train_loader, val_loader, device, epochs=50):

    base_model.eval()  # .to(device)
    policynet.train()  # .to(device)

    optimizer = optim.Adam(policynet.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    losses1, adv_losses = [], []
    losses_en = []
    rewards_batch = []
    acc_batch = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            probs = policynet(images)
            # print('policynet output is',probs)
            # probs = probs * args.alpha + (1 - probs) * (1 - args.alpha)
            distr = Bernoulli(probs)
            policy = distr.sample()
            preds_sample = base_model((images, policy))
            _, predicted = preds_sample.max(1)
            correct += predicted.eq(labels).sum().item()

            ############################### loss function
            reward_sample, match = get_reward(args, preds_sample, labels, policy)
            advantage = reward_sample
            ###acc and drop loss
            log_probs = distr.log_prob(policy)
            log_probs = torch.clamp(log_probs, min=-10.0, max=0.0) ####
            loss1 = -log_probs * advantage.expand_as(policy)
            loss1 = loss1.sum()
            ########entropy_loss
            probs = probs.clamp(1e-15, 1 - 1e-15)
            entropy_loss = probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)
            entropy_loss = args.entropy_weights * entropy_loss.sum()
            batch_acc = match.mean()
            # print('reward loss:{:.3f}, ent_loss:{:.3f}, bat_acc:{:.3f}'.format(loss1,entropy_loss,batch_acc))
            losses1.append(loss1 / images.size(0))
            losses_en.append(entropy_loss / images.size(0))
            acc_batch.append(batch_acc)
            loss = (loss1 + entropy_loss) / images.size(0)  # # /

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # matches.append(match)
            # rewards.append(reward_sample)
            # rewards_batch.append(reward_sample.mean())
            # policies.append(policy.data)
            running_loss += loss.item() * images.size(0)

        train_acc = 100. * correct / total
        val_acc = validate(base_model, policynet, args, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), f'save_models/{args.model}_{args.dataset}_best_model.pth')
            torch.save({
                'model_state_dict': base_model.state_dict(),
                # 'unaveraged_model_state_dict': self.base_model.state_dict(),
                'policynet_state_dict': policynet.state_dict()
            }, f'save_models/{args.model}_{args.dataset}_LeapNet_policy.pth')

def finetune(base_model, policynet, args, train_loader, val_loader, device, epochs=50):

    base_model.eval()  # .to(device)
    policynet.train()  # .to(device)

    checkpoint = torch.load(f'save_models/{args.model}_{args.dataset}_LeapNet_policy.pth', map_location=args.device)
    policynet.load_state_dict(checkpoint['policynet_state_dict'])
    base_model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.SGD(
        base_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            probs = policynet(images)
            # probs = probs * self.params.alpha + (1 - probs) * (1 - self.params.alpha)
            distr = Bernoulli(probs)
            policy = distr.sample()
            preds_sample = base_model((images, policy))
            ##loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(preds_sample, labels)
            # class_losses.append(class_loss)
            # loss = class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # #### clean loss
            # reward_sample, match = get_reward(args, preds_sample, labels, policy)
            # batch_acc = match.mean()
            # # print('reward loss:{:.3f}, ent_loss:{:.3f}, bat_acc:{:.3f}'.format(loss1,entropy_loss,batch_acc))


            running_loss += loss.item() * images.size(0)
            _, predicted = preds_sample.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        val_acc = validate(base_model, policynet, args, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), f'save_models/{args.model}_{args.dataset}_best_model.pth')
            torch.save({
                'model_state_dict': base_model.state_dict(),
                # 'unaveraged_model_state_dict': self.base_model.state_dict(),
                'policynet_state_dict': policynet.state_dict()
            }, f'save_models/{args.model}_{args.dataset}_best_LeapNet_1.pth')

def validate_pretrain(basemodel, args, val_loader, device):
    basemodel.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
            if batch_idx >= 10:
                break
            images, labels = images.to(device), labels.to(device)
            policy = torch.ones(images.size(0), args.num_blocks, dtype=torch.int64).to(device)
            outputs = basemodel((images,policy))
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    acc = 100. * correct / total
    # print(f"Validation Accuracy: {acc:.2f}%")
    return acc

def validate(basemodel, policynet, args, val_loader, device):
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
    # print(f"Validation Accuracy: {acc:.2f}%")
    return acc

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
    # parser.add_argument('--data_dir', type=str, default='resnet18', help='Path to dataset folder (e.g., tiny_imagenet)')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture name (e.g., resnet18, efficientnet-b0)')
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['kul', 'gtsrb', 'cifar10','tiny_imagenet'],
                        help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_validation', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda" or "cpu")')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize input')
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'base', 'cutout', 'autoaugment', 'randaugment', 'idbh'],
                        help='Augment training set.')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--num_blocks', type=int, default=8, help='number of blocks for target model')
    parser.add_argument('--input_size', default=32, type=int, help='input images size')
    parser.add_argument('--num_classes', default=43, type=int, help='number of classes')
    parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
    parser.add_argument('--model_path', type=str, default='save_models/efficientnet_b0_tiny_imagenet_best_model.pth',
                        help='Path to saved model weights')
    parser.add_argument('--save_path', type=str, default='save_models',
                        help='Path to saved model weights')

    parser.add_argument('--policy', default=True, help='Whether to use policy model (skip_resnet)')
    parser.add_argument('--entropy_weights', type=float, default=5, help='entropy multiplier for rl')
    parser.add_argument('--penalty', type=float, default=-10, help='gamma: reward for incorrect predictions')
    parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')

    args = parser.parse_args()

    train_loader, val_loader = load_dataset(args)
    args.num_classes, args.input_size, _ = get_data_info(args)

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
        print('do not know the number of the blocks')#(int(args.model.split('-')[1]) - 4) // 6 * 3
    print('number of blocks is ', args.num_blocks)

    # model = create_model(args.model, num_classes = num_class, device=args.device, policy=args.policy)
    model = create_model(args.model, num_classes=args.num_classes, device=args.device, policy=args.policy)
    policynet = create_rlmodel(args, args.num_blocks, args.device)
    if args.model == 'mobilenet':
        train(model, args, train_loader, val_loader, args.device, epochs=20)
    model.load_state_dict(
        torch.load(f'save_models/{args.model}_{args.dataset}_best_pretrained.pth', map_location=args.device))


    val_acc = validate_pretrain(model, args, val_loader, args.device)
    print(f"Val Acc: {val_acc:.2f}%")

    train_policy(model, policynet, args, train_loader, val_loader, args.device, 10)
    finetune(model, policynet, args, train_loader, val_loader, args.device, args.epochs-10)
