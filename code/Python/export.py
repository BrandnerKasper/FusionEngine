import torch
import torch.nn as nn
from models.pnn import PNN
from models.cnn import CNN
from models.mlp import MLP


def get_model(file_name: str) -> nn.Module:
    if "pnn" in file_name.lower():
        return PNN(4)
    if "cnn" in file_name.lower():
        return CNN(4)
    if "mlp" in file_name.lower():
        return MLP(4)
    raise ValueError(f"{file_name} does not have a model associated with it!")


def export(file_name: str) -> None:
    model = get_model(file_name)
    state = torch.load(f"trained/pth/{file_name}.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(f"trained/pt/{file_name}.pt")


def main() -> None:
    model_file = "CNN_W"
    export(model_file)


if __name__ == '__main__':
    main()
