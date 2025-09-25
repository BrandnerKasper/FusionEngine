import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as FV

from cnn import CNN
from dataloader import ASCIISnake
from utility import grid_to_ascii, one_hot_grid_to_ascii


def test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hot_encode = True
    in_cha = 4 if hot_encode else 1
    model = CNN(in_cha).to(device)

    state = torch.load("pretrained_models/CNN.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    num_workers = 8

    test_dataset = ASCIISnake("data/test", 100, hot_encode=hot_encode)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    for in_txt, gt_img in test_loader:
        in_txt, gt_img = in_txt.squeeze(0).to(device), gt_img.squeeze(0).to(device)
        with torch.inference_mode():
            pred = model(in_txt)
        display(in_txt, pred, gt_img, hot_encode)
        print("Test!")


def display(txt_t: torch.Tensor, pred_t: torch.Tensor, gt_t: torch.Tensor, hot_encode: bool) -> None:
    ascii_str = one_hot_grid_to_ascii(txt_t) if hot_encode else grid_to_ascii(txt_t)
    pred_img = FV.to_pil_image(pred_t)
    gt_img = FV.to_pil_image(gt_t)

    fig, axes = plt.subplots(1, 3, figsize=(8, 9))

    # Left: ASCII text
    axes[0].axis("off")
    axes[0].text(
        0.5, 0.5,
        ascii_str,
        ha="center", va="center",
        family="monospace",
        fontsize=8
    )
    axes[0].set_title("ASCII Grid")

    # Middle: Prediction image
    axes[1].imshow(pred_img)
    axes[1].axis("off")
    axes[1].set_title("Prediction")

    # Right: GT image
    axes[2].imshow(gt_img)
    axes[2].axis("off")
    axes[2].set_title("GT")

    plt.tight_layout()
    plt.show()


def main() -> None:
    test()


if __name__ == '__main__':
    main()
