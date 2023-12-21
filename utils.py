import torch
import torchvision
from dataset import KvasirDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    mask_dir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = KvasirDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,  # Use the same mask_dir for training as images and masks have the same name
        transform=train_transform,
    )

    # Print dataset size information
    print("Train dataset size:", len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = KvasirDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    # Print dataset size information
    print("Validation dataset size:", len(val_ds))

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader



# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device).unsqueeze(1)
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
            
#             # Dice coefficient (F1 score) is intersection of predicted and ground truth regions divided by average size of region 
#             # 2 * (X INT Y) / X + Y
#             dice_score += (2 * (preds * y).sum()) / (
#                 (preds + y).sum() + 1e-8
#             )

#     print(
#         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
#     )
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.train()


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    # Calculate Dice score over the entire dataset after the loop
    intersection = (2 * (preds * y).sum()).float()
    union = (preds + y).sum().float()
    dice_score += intersection / (union + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()



# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

#     model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            # Remove thresholding if predictions are probabilities
            preds = (preds > 0.7).float() # 0.65, .75, .95
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()
