import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio

from tqdm import tqdm
import matplotlib.pyplot as plt

from mlp import MLP
from cnn import CNN
from dataloader import ASCIISnake, grid_to_ascii
from utility import one_hot_grid_to_ascii

torch.manual_seed(42)


def save_model(filename: str, model: nn.Module) -> None:
    model_path = "pretrained_models/" + filename + ".pth"
    torch.save(model.state_dict(), model_path)


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hot_encode = True
    in_cha = 4 if hot_encode else 1
    # model = CNN(in_cha).to(device)
    model = MLP(in_cha).to(device)

    # Hyperparameters
    num_workers = 8
    learning_rate = 0.001
    epochs = 100
    batch_size = 64
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fct = nn.L1Loss()

    # Data
    train_dataset = ASCIISnake("data/train", 1000, hot_encode=hot_encode)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataset = ASCIISnake("data/val", 200, hot_encode=hot_encode)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=num_workers, shuffle=True)

    # Writer
    writer = SummaryWriter()

    # Metrics
    psnr = PeakSignalNoiseRatio().to(device)

    # Loop
    for epoch in tqdm(range(epochs), desc='Train', dynamic_ncols=True):
        total_loss = 0.0

        # Train
        model.train()
        for in_txt, gt_img in tqdm(train_loader, desc=f'Training, Epoch {epoch+1}/{epochs}', dynamic_ncols=True):
            in_txt, gt_img = in_txt.to(device), gt_img.to(device)

            pred = model(in_txt)
            loss = loss_fct(pred, gt_img)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        # Loss
        average_loss = total_loss / len(train_loader)
        print("\n")
        print(f"Loss: {average_loss:.4f}\n")
        writer.add_scalar('Met/Loss', average_loss, epoch)

        # Validation
        if (epoch+1) % 10 != 0:
            continue
        model.eval()
        do_once = True

        for in_txt, gt_img in tqdm(val_loader, desc=f'Val, Epoch {epoch+1}/{epochs}', dynamic_ncols=True):
            in_txt, gt_img = in_txt.to(device), gt_img.to(device)
            with torch.inference_mode():
                pred = model(in_txt)
            pred = torch.clamp(pred, min=0.0, max=1.0)
            psnr.update(pred, gt_img)

            if do_once:
                if hot_encode:
                    ascii_str = one_hot_grid_to_ascii(in_txt[0].detach().cpu())
                else:
                    ascii_str = grid_to_ascii(in_txt[0].detach().cpu)
                fig, axes = plt.subplots(1, 1, figsize=(3, 6))
                axes.axis("off")
                axes.text(
                    0.5, 0.5, ascii_str,
                    ha="center", va="center",
                    family="monospace", fontsize=8
                )
                axes.set_title("ASCII Grid")
                writer.add_figure("Model/In", fig, epoch)
                plt.close(fig)
                writer.add_image("Model/Out", pred[0].detach().cpu(), epoch)
                writer.add_image("Model/GT", gt_img[0].detach().cpu(), epoch)
                do_once = False

        average_psnr = psnr.compute().item()
        psnr.reset()
        print("\n")
        print(f"PSNR {average_psnr:.2f}")
        writer.add_scalar("Met/PSNR", average_psnr, epoch)

    # End
    writer.close()
    save_model("MLP", model)


def main() -> None:
    train()


if __name__ == '__main__':
    main()