import os
import torch
from torch.utils.data import Dataset

from utility import load_txt, load_img, load_txt_one_hot_encode, plot_ascii_img


class ASCIISnake(Dataset):
    def __init__(self, root: str, data_samples: int, hot_encode: bool = False) -> None:
        self.root_in = os.path.join(root, "in")
        self.root_out = os.path.join(root, "out")
        self.data_samples = data_samples
        self.hot_encode = hot_encode
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

        txt = load_txt_one_hot_encode(in_path) if self.hot_encode else load_txt(in_path)
        img = load_img(out_path)

        return txt, img

    def plot(self, idx: int) -> None:
        txt_tensor, img_tensor = self[idx]
        plot_ascii_img(txt_tensor, img_tensor, self.hot_encode)


def main() -> None:
    path = "data/test"
    ascii_snake_data = ASCIISnake(path, 1000, False)
    ascii_snake_data.plot(0)


if __name__ == '__main__':
    main()
