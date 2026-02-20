import argparse
import csv

import torch
from torch.utils.data import DataLoader
import yaml

from dataset import CIFAR10Dataset
from model import SimpleCNN
from train import train_one_epoch, evaluate


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_params(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def append_to_csv(filename, row_dict):
    file_exists = False
    try:
        with open(filename, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="exp1, exp2, or exp3")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    exp = params["experiments"][args.exp]

    # General settings
    seed = params.get("seed", 42)
    torch.manual_seed(seed)

    data_root = params["data"]["root"]
    num_workers = params["data"].get("num_workers", 2)

    epochs = params["train"]["epochs"]
    weight_decay = params["train"].get("weight_decay", 0.0)

    batch_size = exp["batch_size"]
    lr = exp["lr"]

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("CUDA build:", torch.version.cuda)

    # Dataset + DataLoader
    train_ds = CIFAR10Dataset(root=data_root, train=True)
    test_ds = CIFAR10Dataset(root=data_root, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc

    # Save results to a simple CSV for your README table
    append_to_csv(
        "runs.csv",
        {
            "experiment": args.exp,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "best_val_acc": best_acc,
            "device": str(device),
        },
    )

    print("\nDone ✅")
    print(f"Best val acc: {best_acc:.4f}")
    print("Saved results to runs.csv")


if __name__ == "__main__":
    main()