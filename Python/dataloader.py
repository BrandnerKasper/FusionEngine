import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as FV


class ASCIISnake(Dataset):
    def __init__(self, root: str, data_samples: int) -> None:
        self.root_in = os.path.join(root, "in")
        self.root_out = os.path.join(root, "out")
        self.data_samples = data_samples
        self.filenames = self.init_filenames()

    def init_filenames(self) -> list[str]:
        filenames = []
        for i in range(0, self.data_samples):
            filenames.append(f"{i:0{4}d}")
        return sorted(set(filenames))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        in_path = os.path.join(self.root_in, filename)
        out_path = os.path.join(self.root_out, filename)

        txt = load_txt(in_path)
        img = load_img(out_path)

        return txt, img

    def display_item(self, idx: int) -> None:
        txt_tensor, img_tensor = self[idx]

        ascii_str = tensor_to_ascii(txt_tensor)
        img = FV.to_pil_image(img_tensor)

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

# utility
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
    return torch.tensor(grid, dtype=torch.float32)


def load_img(path: str) -> torch.Tensor:
    img = cv2.imread(f"{path}.png", cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    return transform(img)


id_to_char = {v: k for k, v in char_map.items()}

def tensor_to_ascii(grid: torch.Tensor) -> str:
    lines = []
    for row in grid.tolist():
        chars = [id_to_char.get(val, '?') for val in row]
        lines.append(''.join(chars))
    return '\n'.join(lines)


def main() -> None:
    path = "data/test"
    ascii_snake_data = ASCIISnake(path, 1000)
    ascii_snake_data.display_item(0)


if __name__ == '__main__':
    main()
