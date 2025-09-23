import torch
import cv2
from torchvision import transforms


# Utility fct. to turn txt, img files into tensors and back

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


def load_img(path: str) -> torch.Tensor:
    img = cv2.imread(f"{path}.png", cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    return transform(img)


id_to_char = {v: k for k, v in char_map.items()}

def tensor_to_ascii(grid: torch.Tensor) -> str:
    lines = []
    for row in grid.squeeze(0).tolist():
        chars = [id_to_char.get(val, '?') for val in row]
        lines.append(''.join(chars))
    return '\n'.join(lines)