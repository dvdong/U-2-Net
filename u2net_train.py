import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP


# ------- 1. define loss function --------

# Binary Cross Entropy (BCE) loss function, commonly used for binary classification tasks.
bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """
    Compute the multi-scale BCE loss for the U2-Net model outputs.
    Each output (d0 to d6) corresponds to a different scale of the prediction.

    Args:
        d0, d1, ..., d6: Model outputs at different scales.
        labels_v: Ground truth labels.

    Returns:
        loss0: BCE loss for the main output (d0).
        loss: Combined loss for all outputs.
    """
    # Compute BCE loss for each scale
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    # Sum up all losses
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    # Print individual losses for debugging
    print(
        "l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"
        % (
            loss0.data.item(),
            loss1.data.item(),
            loss2.data.item(),
            loss3.data.item(),
            loss4.data.item(),
            loss5.data.item(),
            loss6.data.item(),
        )
    )

    return loss0, loss


# ------- 2. set the directory of training dataset --------

# Model name, can be either 'u2net' or 'u2netp' (a lightweight version of U2-Net).
model_name = "u2net"

# Paths for dataset and model saving
data_dir = os.path.join(os.getcwd(), "new_train_data" + os.sep)  # Dataset folder
tra_image_dir = os.path.join("images" + os.sep)  # Training images folder
tra_label_dir = os.path.join("masks" + os.sep)  # Training labels folder

image_ext = ".jpg"  # Image file extension
label_ext = ".png"  # Label file extension

model_dir = os.path.join(os.getcwd(), "saved_models", model_name + os.sep)  # Model save folder

# Training parameters
epoch_num = 1000  # Number of training epochs
batch_size_train = 24  # Batch size for training
batch_size_val = 1  # Batch size for validation

train_num = 0
val_num = 0

# Get the list of training images and corresponding labels
train_image_list = glob.glob(data_dir + tra_image_dir + "*" + image_ext)  # List of training image paths
train_label_list = []  # List of training label paths

# Match each image with its corresponding label
for image_path in train_image_list:
    image_name = image_path.split(os.sep)[-1]  # Extract file name from path

    name_parts = image_name.split(".")  # Split file name by "."
    name_without_ext = name_parts[0:-1]  # Remove file extension
    label_name = name_without_ext[0]  # Initialize label name
    for i in range(1, len(name_without_ext)):  # Reconstruct name if multiple "." exist
        label_name = label_name + "." + name_without_ext[i]

    # Append the corresponding label path
    train_label_list.append(data_dir + tra_label_dir + label_name + label_ext)

# Print dataset statistics
print("---")
print("train images: ", len(train_image_list))
print("train labels: ", len(train_label_list))
print("---")


# ------- start training --------

def train():
    """
    Main training function for the U2-Net model.
    """
    train_num = len(train_image_list)  # Number of training images

    # Create the dataset and dataloader
    salobj_dataset = SalObjDataset(
        img_name_list=train_image_list,
        lbl_name_list=train_label_list,
        transform=transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]), # 对输入图像进行缩放、裁剪和转换为张量
    )

    salobj_dataloader = DataLoader(
        salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1
    )

    # ------- 3. define model --------
    # Initialize the model based on the selected model name
    if model_name == "u2net":
        net = U2NET(3, 1)  # U2-Net model
    elif model_name == "u2netp":
        net = U2NETP(3, 1)  # Lightweight U2-Net model

    # Move the model to GPU if available
    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------

    print("---define optimizer...")
    # 创建一个 Adam 优化器 的，用于训练神经网络 
    optimizer = optim.Adam(
        net.parameters(), # 要优化的参数（即网络中所有需要训练的权重）
        lr=1e-5, # 学习率（Learning Rate）
        betas=(0.9, 0.999), # Adam 特有的两个动量参数
        eps=1e-08,  # 防止除以0的小常数，增强数值稳定性
        weight_decay=0 # 权重衰减（L2正则化），这里为0表示不使用
    )

    # ------- 5. training process --------

    print("---start training...")
    ite_num = 0  # Total iteration count
    running_loss = 0.0  # Accumulated loss
    running_tar_loss = 0.0  # Accumulated target loss
    ite_num4val = 0  # Iteration count for validation
    save_frq = 2000  # Save the model every 2000 iterations

    for epoch in range(0, epoch_num):  # Loop over epochs
        net.train()  # Set model to training mode

        for i, data in enumerate(salobj_dataloader):  # Loop over batches
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data["image"], data["label"]  # Get inputs and labels

            inputs = inputs.type(torch.FloatTensor)  # Convert inputs to float tensor
            labels = labels.type(torch.FloatTensor)  # Convert labels to float tensor

            # Wrap inputs and labels in Variables
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, compute loss, and backpropagate
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)  # Model outputs

            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)  # Compute loss

            loss.backward()  # Backpropagation

            optimizer.step()  # Update weights

            # Accumulate losses
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # Delete temporary variables to free memory
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            # Print training statistics
            print(
                "[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f "
                % (
                    epoch + 1,
                    epoch_num,
                    (i + 1) * batch_size_train,
                    train_num,
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val,
                )
            )

            # Save the model at specified intervals
            if ite_num % save_frq == 0:
                torch.save(
                    net.state_dict(),
                    model_dir
                    + model_name
                    + "_bce_itr_%d_train_%3f_tar_%3f.pth"
                    % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val),
                )
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # Resume training
                ite_num4val = 0

if __name__ == "__main__":
    train()  # Start training
