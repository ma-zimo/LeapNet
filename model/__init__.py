import torch
import torch.nn as nn
from .resnet import Normalization
from .resnet import resnet
from .skip_resnet import skip_resnet
from .skip_resnet_inf import skip_resnet_inf
from .efficienetnet import efficientnet
from .skip_efficientnet import skip_efficientnet
from .policynet import small
from .policynet import PolicyNet
from .policynet import PolicyNet_L
from .wideresnet import wideresnet
# from .efficientnet import efficientnet
from .mobilenet import mobilenet
from .skip_mobilenet_inf import skip_mobilenet_inf
from .skip_mobilenet import skip_mobilenet
from .latency_predictor import Predictor
from .skip_resnet import PolicyNet_oldversion

def create_model(name, num_classes, device, policy=False, pretrained=True, inference=False):
    if inference == False:
        """during inference, policynet control directly skip useless layers for less computation, 
    but during training, we still need to compute these skipped layers and multiply zero for skipped layers because of batch processing.
    Sepcially, in a batch, each inference has different inference path (skipping different layers)"""
        #### whether "inference" is true or false only affects latency performance
        if 'resnet' in name:
            if policy is False:
                backbone = resnet(name, num_classes=num_classes, pretrained=pretrained)
            else:
                backbone = skip_resnet(name, num_classes=num_classes, pretrained=pretrained)
        elif 'efficientnet' in name:
            if policy is False:
                backbone = efficientnet(name, num_classes=num_classes)
            else:
                backbone = skip_efficientnet(name, num_classes=num_classes)
        elif 'mobilenet' in name:
            if policy is False:
                backbone = mobilenet(num_classes=num_classes)
            else:
                backbone = skip_mobilenet(num_classes=num_classes)
        else:
            raise ValueError(f"Model '{name}' not supported.")
    else:
        if 'resnet' in name:
            if policy is False:
                backbone = resnet(name, num_classes=num_classes, pretrained=pretrained)
            else:
                backbone = skip_resnet_inf(name, num_classes=num_classes, pretrained=pretrained)
        elif 'mobilenet' in name:
            if policy is False:
                backbone = mobilenet(num_classes=num_classes)
            else:
                backbone = skip_mobilenet_inf(num_classes=num_classes)
        else:
            raise ValueError(f"Model '{name}' not supported.")

    model = backbone #torch.nn.Sequential(backbone)
    return model.to(device)

def create_rlmodel(args, num_blocks, device, latency=False):
    """
       Returns suitable model from its name.
       Arguments:
           name (str): name of resnet architecture.
           normalize (bool): normalize input.
           info (dict): dataset information.
           device (str or torch.device): device to work on.
       Returns:
           torch.nn.Module.
       """
    if latency == False:
        if args.model == 'resnet18' or args.model == 'resnet34':
            backbone = PolicyNet_oldversion(args, num_blocks)
            ###since the new version weights are not found, we found a old version weights with similar peformance.
        else:
            backbone = PolicyNet(args, num_blocks)
            # print("create 'LeapNet-1's decision network")
    else:
        backbone = PolicyNet_L(args, num_blocks)

    # model = torch.nn.Sequential(backbone)

    model = backbone.to(device)
    return model

def create_latpredictor(device, num_blocks):
    # print('number of blocks is ', num_blocks)
    model = Predictor(num_blocks)
    model = model.to(device)
    return model
