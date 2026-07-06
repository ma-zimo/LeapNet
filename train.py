import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model
from dataset import load_dataset
import argparse
import os
from tqdm import tqdm
import torchattacks
import random

def train(model, train_loader, val_loader, device, epochs=50):
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

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'save_models/{args.model}_{args.dataset}_best_pretrained.pth')

def train_adv(model, train_loader, val_loader, device, epochs=50):
    # optimizer = optim.SGD([
    #     {"params": model.features.parameters(), "lr": 1e-4},
    #     {"params": model.classifier.parameters(), "lr": 1e-3},
    # ], momentum=0.9, weight_decay=5e-4)

    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )

    # 学习率调度器：每 10 epoch 衰减一次学习率，衰减因子为 0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_acc = 0.0
    best_acc_adv = 0.0

    # epsilon = 0.03
    # atk = torchattacks.FGSM(model, eps=epsilon)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, correct_adv, total = 0, 0, 0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_clean = criterion(outputs, labels)

            model.eval()  # 攻击期间需关闭 BN 和 Dropout
            epsilon = random.uniform(0.0, 0.05)
            atk = torchattacks.FGSM(model, eps=epsilon)
            adv_images = atk(images, labels)
            model.train()
            outputs_adv = model(adv_images)
            loss_adv = criterion(outputs_adv, labels)

            loss = loss_clean + 0.5*loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            _, predicted_adv = outputs_adv.max(1)
            correct_adv += predicted_adv.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        train_acc_adv = 100. * correct_adv / total
        val_acc = validate(model, val_loader, device)
        val_acc_adv = evaluate_adv(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f}, "
              f"Train Clean Acc: {train_acc:.2f}%, Train Adv Acc: {train_acc_adv:.2f}%"
              f"Val Acc: {val_acc:.2f}%, Val Adv Acc: {val_acc_adv:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'save_models/{args.model}_{args.dataset}_best_model_{best_acc}.pth')

        if val_acc_adv > best_acc_adv:
            best_acc_adv = val_acc_adv
            torch.save(model.state_dict(), f'save_models/{args.model}_{args.dataset}_best_adv_model_{best_acc_adv}.pth')

def validate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
            if batch_idx >= 20:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def evaluate_adv(model, val_loader, device):

    model.eval()
    correct = 0
    total = 0
    epsilon = 0.03

    # print('attack is FGSM, epsilon is ', epsilon)
    # attack = torchattacks.BIM(model, eps=epsilon, alpha=2 / 255, steps=10)
    # attack = torchattacks.DeepFool(self.model, steps=step, overshoot=overshoot)
    attack = torchattacks.FGSM(model, eps=epsilon)
    # attack = torchattacks.Square(model, norm='Linf', eps=epsilon, n_queries=5000, n_restarts=1,
    #                              p_init=.8, seed=0, verbose=False, loss='margin',
    #                              resc_schedule=True)
    # attack = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, version='standard', n_classes=43,
    #                                  seed=None,
    #                                  verbose=False)

    for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
        if batch_idx >= 20:
            break
        images = images.to(device)
        labels = labels.to(device)
        adv_images = attack(images, labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100. * correct / total
    # print(f"Test Adv Accuracy: {acc*100:.2f}%")

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
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_size_validation', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda" or "cpu")')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize input')
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'base', 'cutout', 'autoaugment', 'randaugment', 'idbh'],
                        help='Augment training set.')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--model_path', type=str, default='save_models/best_model.pth',
                        help='Path to saved model weights')

    args = parser.parse_args()

    train_loader, val_loader = load_dataset(args)
    num_class, input_size, _ = get_data_info(args)
    # model = create_model(args.model, num_classes = num_class, device=args.device, policy=args.policy)
    model = create_model(args.model, num_classes=num_class, device=args.device, pretrained=False)

    ######## normal training
    train(model, train_loader, val_loader, args.device, args.epochs)

    ######## adversarial training
    # checkpoint = torch.load(args.model_path, map_location=args.device)
    # state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint)
    # train_adv(model, train_loader, val_loader, args.device, args.epochs)
