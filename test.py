import torch
import argparse
from torch.utils.data import DataLoader
from model_unet import U_Net  # Assuming this module exists
from test_Dataset import MyTestDataset  # Assuming you have a dataset module
import os
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch EIVIM")
parser.add_argument("--testdir", default="/homes/lwjiang/Data/IVIM/val_bymyself/")
parser.add_argument("--targetdir", default="/homes/lwjiang/Data/IVIM/pred_val_bymyself/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming model is saved with torch.save(model.state_dict(), PATH)
model = U_Net(in_ch=8, out_ch=4).to(device)  # Initialize your model
model.load_state_dict(torch.load('/homes/lwjiang/Data/IVIM/public_training_data/net/U_net.pkl'))
model.eval()


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)
    test_dir = opt.testdir
    target_dir = opt.targetdir
    
    test_dataset = MyTestDataset(test_dir, transform=None)  # Initialize your dataset
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Define batch size

    preds = []
    with torch.no_grad():
        for step, batch_data in enumerate(test_dataloader):
            noisy_image, _ = batch_data
            noisy_image = noisy_image.to(device)
            output = model(noisy_image)
            out = output[0, :3, :, :]    # 深度图的形状: (1, 4, 200, 200)，去掉第一维batch，和生成的b0
            out = out.permute(1, 2, 0)

            filename = "{:04}".format(step+1) + "_IVIMParam"
            de_dir = os.path.join(target_dir , filename)
            np.save(de_dir, out.cpu().numpy())


    # Do something with predictions

if __name__ == '__main__':
    main()