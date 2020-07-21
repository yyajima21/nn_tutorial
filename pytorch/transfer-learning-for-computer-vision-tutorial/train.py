# License: BSD
# Original Author: Sasank Chilamkurthy
# This code is originally developed by Sasank Chilamkurthy 
# from the following website: 
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# I modified and refactoed the original code for my learning purpose.

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import copy

import utils


class Setting:
    def __init__(self):
        self.data_dir = "data/hymenoptera_data"
        self.args = utils.get_args()
        self.data_transforms = self.data_transforms()
        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.data_dir, x), self.data_transforms[x]
            )
            for x in ["train", "val"]
        }
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x],
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
            )
            for x in ["train", "val"]
        }
        self.dataset_sizes = {
            x: len(self.image_datasets[x]) for x in ["train", "val"]
        }
        self.class_names = self.image_datasets["train"].classes
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def data_transforms(self):
        # Data augmentation and normalization for training
        # # Just normalization for validation
        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }
        return self.data_transforms


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(1)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, setting):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(setting.args.epochs):
        print("Epoch {}/{}".format(epoch, setting.args.epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in setting.dataloaders[phase]:
                inputs = inputs.to(setting.device)
                labels = labels.to(setting.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / setting.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / setting.dataset_sizes[phase]

            print(
                "{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, setting, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(setting.dataloaders["val"]):
            inputs = inputs.to(setting.device)
            labels = labels.to(setting.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(
                    "predicted: {}".format(setting.class_names[preds[j]])
                )
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def main():
    # the function to turn on interactive mode
    plt.ion()
    setting = Setting()

    if setting.args.check_images:
        # Get a batch of training data
        inputs, classes = next(iter(setting.dataloaders["train"]))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=[setting.class_names[x] for x in classes])

    if setting.args.mode == "fixed_feature_extractor":
        print("mode: {}".format(setting.args.mode))
        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)

        model_conv = model_conv.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        optimizer_conv = optim.SGD(
            model_conv.fc.parameters(),
            lr=setting.args.lr,
            momentum=setting.args.momentum,
        )

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_conv, step_size=7, gamma=0.1
        )

        model_conv = train_model(
            model_conv, criterion, optimizer_conv, exp_lr_scheduler, setting
        )

        visualize_model(model_conv, setting)

        plt.ioff()
        plt.show()

        if setting.args.save_model:
            torch.save(
                model_conv.state_dict(), "fixed_feature_extractor_cnn.pt"
            )

    if setting.args.mode == "finetuning_convnet":
        print("mode: {}".format(setting.args.mode))
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features

        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(setting.device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(
            model_ft.parameters(),
            lr=setting.args.lr,
            momentum=setting.args.momentum,
        )

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=0.1
        )

        model_ft = train_model(
            model_ft, criterion, optimizer_ft, exp_lr_scheduler, setting
        )

        visualize_model(model_ft, setting)
        if setting.args.save_model:
            torch.save(model_ft.state_dict(), "finetuning_convnet_cnn.pt")


if __name__ == "__main__":
    main()
