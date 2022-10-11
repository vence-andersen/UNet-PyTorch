from operator import ge
from pickletools import optimize
from turtle import backward
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
# from unet import UNet
from unet_aladin import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_images
)

# Hyperparameters
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
Batch_size = 4
epochs = 3
num_workers = 2
image_height = 572
image_width = 572
pin_memory = True
load_model=  False
train_image_dir = "data/train_images/"
train_mask_dir = "data/train_mask/"
val_image_dir = "data/val_images/"
val_mask_dir = "data/val_mask/"

def train_fn(loader, model, optimiser, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            targets = targets.float().unsqueeze(1).to(device=device)
            loss = loss_fn(predictions, targets)

        # backward
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # update tqdm loop

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                    mean = [0.0, 0.0, 0.0],
                    std = [1.0, 1.0, 1.0],
                    max_pixel_value = 255.0,
                ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader = get_loaders(
        train_image_dir, train_mask_dir,
        val_image_dir, val_mask_dir,
        Batch_size, num_workers,
        train_transform, val_transform,
        pin_memory
    )

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        train_fn(train_loader, model, optimiser, loss_fn, scaler)

        # save_model
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimiser" : optimiser.state_dict()
        }
        save_checkpoint(checkpoint)

        # check_accuracy
        check_accuracy(val_loader, model, device=device)

        #saving output
        save_predictions_as_images(val_loader, model, folder="saved_images/", device="cuda")


if __name__ == "__main__":
    main()