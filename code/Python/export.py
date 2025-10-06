import torch
from models.pnn import PNN


def main() -> None:
    model = PNN(4)
    state = torch.load("pretrained_models/PNN_W.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save("pretrained_models/PNN_W.pt")


if __name__ == '__main__':
    main()
