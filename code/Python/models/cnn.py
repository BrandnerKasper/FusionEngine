import torch
import torch.nn as nn

from utility import summary, measure_inference, measure_vram_usage


class CNN(nn.Module):
    def __init__(self, in_cha: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_cha, 32, 1, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 32, 1, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch. Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(4).to(device)
    batch_size = 1
    input_data = (batch_size, 4, 32, 32)

    summary(model, input_data)
    measure_inference(model, input_data)
    measure_vram_usage(model, input_data)


if __name__ == "__main__":
    main()
