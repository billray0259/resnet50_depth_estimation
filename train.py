import os

while "src" not in os.listdir():
    assert "/" != os.getcwd(), "src directory not found"
    os.chdir("..")

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.lib.nyu_dataset import NYUDataset, transform
from src.lib.depth_estimator import DepthEstimator
from src.lib.resnet_loader import load_classifier_resnet50, load_contrastive_resnet50



# Parse command line arguments
parser = argparse.ArgumentParser()

# classification
parser.add_argument("--pretrain_type", type=str, default="classification", help="classification or contrastive")
# probing
parser.add_argument("--probing", type=bool, default=False, help="Whether or not to finetune the reset encoder")
# switch
parser.add_argument("--switch", type=bool, default=False, help="Whether or not to switch from probing to finetuning. Overrides probing")
# batch size
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
# epochs
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
# learning rate
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
# dropout rate
parser.add_argument("--dropout_rate", type=float, default=0.4, help="Dropout rate")

# Parse arguments
args = parser.parse_args()

assert args.pretrain_type in ["classification", "contrastive"], "pretrain_type must be classification or contrastive"


DATA_DIR = "data"
DATASET_FILE = "nyu_depth_v2_labeled.mat"

# batch_size = args.batch_size
# lr = args.learning_rate
# epochs = args.epochs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset = NYUDataset(os.path.join(DATA_DIR, DATASET_FILE), transform=transform)

n_train, n_val = int(0.8 * len(dataset)), int(0.1 * len(dataset))
n_test = len(dataset) - n_train - n_val

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8)


classification = args.pretrain_type == "classification"

if classification:
    resnet = load_classifier_resnet50()
else:
    resnet = load_contrastive_resnet50("pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar")

model = DepthEstimator(resnet, probing=args.probing, dropout_p=args.dropout_rate).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()


validation_loss_history = []
loss_history = []

for epoch in range(args.epochs):
    if args.switch:
        if epoch < args.epochs//2:
            model.probing = True
        else:
            model.probing = False
        
    epoch_loss = 0
    epoch_loss_count = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    model.train()
    for i, (color_map, depth_map) in pbar:
        color_map = color_map.to(device)
        depth_map = depth_map.to(device)

        pred = model(color_map)
        loss = loss_fn(pred, depth_map)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss_history.append(loss.item())
        epoch_loss += loss.item()
        epoch_loss_count += 1

        # update pbar description with batch and loss
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} - Batch {i + 1}/{len(train_loader)} - Loss: {epoch_loss/epoch_loss_count:.4f}")
    
    loss_history.append(epoch_loss/epoch_loss_count)
    
    # take tensors we don't need for validation off the gpu
    del color_map, depth_map, pred, loss
    torch.cuda.empty_cache()
    
    # compute validation loss
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (color_map, depth_map) in enumerate(val_loader):
            color_map = color_map.to(device)
            depth_map = depth_map.to(device)

            pred = model(color_map)
            val_loss += loss_fn(pred, depth_map).item()
        val_loss /= len(val_loader)

        validation_loss_history.append(val_loss)
    # display validation loss
    print(f"Epoch {epoch + 1}/{args.epochs} - Validation Loss: {val_loss:.4f}")


probing_str = "probing" if args.probing else ("switch" if args.switch else "finetuning")

experiment_name = f"{args.pretrain_type}_{probing_str}_bs{args.batch_size}_lr{args.learning_rate}_epochs{args.epochs}_dropout{args.dropout_rate}"
experiment_dir = os.path.join("experiments", experiment_name)
if os.path.exists(experiment_dir):
    i = 1
    while os.path.exists(experiment_dir + "_" + str(i)):
        i += 1
    experiment_dir += "_" + str(i)

os.mkdir(experiment_dir)

# save model
with open(os.path.join(experiment_dir, "model.pth"), "wb") as f:
    torch.save(model, f)

config = {
    "classification": classification,
    "probing": args.probing,
    "switch": args.switch,
    "batch_size": args.batch_size,
    "lr": args.learning_rate,
    "epochs": args.epochs,
    "dropout_rate": args.dropout_rate,
    "device": device.type,
    "experiment_dir": experiment_dir
}

# save config
with open(os.path.join(experiment_dir, "config.json"), "w") as f:
    json.dump(config, f)

histories = {
    "loss": loss_history,
    "val_loss": validation_loss_history
}

# save histories
with open(os.path.join(experiment_dir, "histories.json"), "w") as f:
    json.dump(histories, f)



# Plot loss history
plt.plot(loss_history)
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training Loss")
# save figure
plt.savefig(os.path.join(experiment_dir, "training_loss_history.png"))
plt.clf()

# Plot validation loss history
plt.plot(validation_loss_history)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss")
# save figure
plt.savefig(os.path.join(experiment_dir, "validation_loss_history.png"))
plt.clf()


# Visualize the models predictions
# pick 9 random samples
idxs = np.random.choice(len(dataset), 9)
samples = [dataset[idx] for idx in idxs]
color_maps = []
depth_maps = []
preds = []
for color_tensor, depth_tensor in samples:
    # to numpy array
    color_map = color_tensor.cpu().numpy().transpose(1, 2, 0)
    depth_map = depth_tensor.cpu().numpy().squeeze()
    depth_prediction = model(color_tensor.unsqueeze(0).to(device)).cpu().detach().squeeze().numpy()

    color_maps.append(color_map)
    depth_maps.append(depth_map)
    preds.append(depth_prediction)

# plot 3x9 grid of color, depth, and prediction
fig, axes = plt.subplots(3, 9, figsize=(9*4, 3*4))
for i in range(9):
    axes[0, i].imshow(color_maps[i])
    depth_map = depth_maps[i]
    # scale between 0 and 1
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    pred = preds[i]
    # scale between 0 and 1
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    axes[2, i].imshow(depth_map)
    axes[1, i].imshow(pred)

for ax in axes.ravel():
    ax.axis("off")

# save figure
plt.savefig(os.path.join(experiment_dir, "predictions.png"))

plt.clf()



