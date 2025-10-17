import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torchvision.transforms import functional as FV
import torch.nn.functional as F
import torchinfo
import matplotlib.pyplot as plt

# Data
char_map = {
    ' ': 0,
    '#': 1,
    'o': 2,
    '•': 3
}


def load_txt(path: str) -> torch.Tensor:
    with open(f"{path}.txt", "r", encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file]
    grid = [[char_map.get(c) for c in line] for line in lines]
    return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)


K = len(char_map)


def load_txt_one_hot_encode(path: str) -> torch.Tensor:
    with open(f"{path}.txt", "r", encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file]
    grid = [[char_map.get(c) for c in line] for line in lines]
    grid = torch.tensor(grid, dtype=torch.long)
    one_hot = F.one_hot(grid, num_classes=K).permute(2, 0, 1).to(torch.float32)
    return one_hot


def load_img(path: str) -> torch.Tensor:
    img = cv2.imread(f"{path}.png", cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    return transform(img)


id_to_char = {v: k for k, v in char_map.items()}


def grid_to_ascii(grid: torch.Tensor) -> str:
    lines = []
    for row in grid.squeeze(0).tolist():
        chars = [id_to_char.get(val, '?') for val in row]
        lines.append(''.join(chars))
    return '\n'.join(lines)


def one_hot_grid_to_ascii(one_hot: torch.Tensor) -> str:
    ids = one_hot.argmax(dim=0)
    lines = [''.join(id_to_char.get(int(v), '?') for v in row) for row in ids.tolist()]
    return '\n'.join(lines)


def plot_ascii_img(txt_t: torch.Tensor, img_t: torch.Tensor, hot_encode: bool) -> None:
    ascii_str = one_hot_grid_to_ascii(txt_t) if hot_encode else grid_to_ascii(txt_t)
    img = FV.to_pil_image(img_t)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

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

    # Right: Image
    axes[1].imshow(img)
    axes[1].axis("off")
    axes[1].set_title("Image")

    plt.tight_layout()
    plt.show()


def plot_ascii_pred_gt(txt_t: torch.Tensor, pred_t: torch.Tensor, gt_t: torch.Tensor, hot_encode: bool) -> None:
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


# Model
def summary(model: nn.Module, input_size: tuple[int, int, int, int]):
    torchinfo.summary(model, input_size)


def measure_inference(model: nn.Module, input_size: tuple[int, int, int, int]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model.to(device)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    input_data = torch.randn(input_size).to(device)

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    print("Start timing ...")
    torch.cuda.synchronize()
    iterations = 10
    with torch.no_grad():
        total = 0
        for i in range(iterations):
            start.record()
            _ = model(input_data)
            end.record()
            torch.cuda.synchronize()
            total += start.elapsed_time(end)
    average = total / iterations
    print(f"Average forward pass time {average:.2f} ms")


def measure_vram_usage(model: nn.Module, input_size: tuple[int, int, int, int]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    # Generate dummy input
    input_data = torch.randn(input_size).to(device)

    with torch.no_grad():
        _ = model(input_data)
    print("Memory allocated (peak):", torch.cuda.max_memory_allocated() / 1024 ** 2, "MB")
