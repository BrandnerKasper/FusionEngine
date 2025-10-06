import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import torch.nn.functional as F
import torchinfo

# Data loading
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
    one_hot = F.one_hot(grid, num_classes=K).permute(2, 0 , 1).to(torch.float32)
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