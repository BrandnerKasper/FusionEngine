import torch
from torch.utils.data import DataLoader

from models.pnn import PNN
from dataloader import ASCIISnake
from utility import plot_ascii_pred_gt


def test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hot_encode = True
    in_cha = 4 if hot_encode else 1
    model = PNN(in_cha).to(device)

    state = torch.load("pretrained_models/CNN_W.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    num_workers = 8

    test_dataset = ASCIISnake("data/test", 100, hot_encode=hot_encode)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    for in_txt, gt_img in test_loader:
        in_txt, gt_img = in_txt.squeeze(0).to(device), gt_img.squeeze(0).to(device)
        with torch.inference_mode():
            pred = model(in_txt)
        plot_ascii_pred_gt(in_txt, pred, gt_img, hot_encode)
        print("Test!")


def main() -> None:
    test()


if __name__ == '__main__':
    main()
