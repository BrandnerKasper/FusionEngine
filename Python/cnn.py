import torch
import torch.nn as nn
from torchinfo import summary


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

    # UTILITY
    def summary(self, input_size) -> None:
        summary(self, input_size)

    def measure_inference(self, input_size) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Move the model to GPU if available
        self.to(device)

        # eval mode
        self.eval()
        for k, v in self.named_parameters():
            v.requires_grad = False

        # Generate dummy input
        input_data = torch.randn(input_size).to(device)

        # GPU warm up
        print("Warm up ...")
        with torch.no_grad():
            for _ in range(10):
                _ = self(input_data)

        print("Start timing ...")
        torch.cuda.synchronize()
        iterations = 10
        with torch.no_grad():
            total = 0
            for i in range(iterations):
                start.record()
                _ = self(input_data)
                end.record()
                torch.cuda.synchronize()
                total += start.elapsed_time(end)
        average = total / iterations
        print(f"Average forward pass time {average:.2f} ms")

    def measure_v_ram_usage(self, input_size) -> None:
        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # eval mode
        self.eval()
        for k, v in self.named_parameters():
            v.requires_grad = False

        # Generate dummy input
        is_tuple_of_tuples = all(map(lambda x: isinstance(x, tuple), input_size))
        if is_tuple_of_tuples:
            input_data = []
            for item in input_size:
                data = torch.randn(item).to(device)
                input_data.append(data)
        else:
            input_data = torch.randn(input_size).to(device)

        with torch.no_grad():
            if is_tuple_of_tuples:
                _ = self(*input_data)
            else:
                _ = self(input_data)
        print("Memory allocated (peak):", torch.cuda.max_memory_allocated() / 1024 ** 2, "MB")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(4).to(device)
    batch_size = 1
    input_data = (batch_size, 4, 32, 32)

    model.summary(input_data)
    model.measure_inference(input_data)
    model.measure_v_ram_usage(input_data)


if __name__ == "__main__":
    main()
