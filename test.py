import torch
import argparse
from model import U_Net

parser = argparse.ArgumentParser(description="PyTorch EIVIM")
parser.add_argument("--testdir", default="/homes/lwjiang/Data/IVIM/public_validation_data/")

model = U_Net
model.load_state_dict(torch.load('U_net.pkl'))
model.eval()


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    test_dir = opt.testdir

    for noisy_image in test_dir:
        with torch.no_grad():
            test_out = model(noisy_image)


if __name__ == '__main__':
    main()