import torch
import torch.nn as nn

from utility import summary, measure_inference, measure_vram_usage


class PNN(nn.Module):
    def __init__(self, k: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(k, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PNN().to(device)
    batch_size = 1
    input_data = (batch_size, 4, 32, 32)

    summary(model, input_data)
    measure_inference(model, input_data)
    measure_vram_usage(model, input_data)


if __name__ == "__main__":
    main()
