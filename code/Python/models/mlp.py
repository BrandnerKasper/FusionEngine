import torch
import torch.nn as nn

from utility import summary, measure_inference, measure_vram_usage


class MLP(nn.Module):
    def __init__(self, in_cha: int = 1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32*in_cha, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 32*32*3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = x.view(x.size(0), 3, 32, 32)
        x = torch.sigmoid(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(4).to(device)
    batch_size = 1
    input_data = (batch_size, 4, 32, 32)

    summary(model, input_data)
    measure_inference(model, input_data)
    measure_vram_usage(model, input_data)


if __name__ == "__main__":
    main()
