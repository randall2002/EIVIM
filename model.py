import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchvision.transforms import CenterCrop

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x
    
class crop_cat(nn.Module):
    def forward(self, x, x_contract):
        x_contract = CenterCrop(x_contract,[x.shape[2],x.shape[3]])
        x_cat = torch.cat([x,x_contract],dim=1)
        return x_cat

class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = conv_block(in_ch, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])
        self.conv4 = conv_block(filters[2], filters[3])
        self.conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.Crop_Cat = crop_cat()

    def forward(self, x):
        e1 = self.conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.Crop_Cat(e4, d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.Crop_Cat(e3, d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Crop_Cat(e2, d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Crop_Cat(e1, d2)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out

#测试代码
from torchsummary import summary
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = U_Net(in_ch= 8, out_ch=3).to(device)
    #print(unet)
    summary(unet, input_size=(8, 256, 256))
    #项目典型数据维度是：(4, 8, 200, 200)，对导致拼接不匹配，需要做相应调整；

    tmp = torch.randn(4, 8, 256, 256).to(device)
    out = unet(tmp)
    print('out.shape:', out.shape)
    p = sum(map(lambda p: p.numel(), unet.parameters()))
    print('parameters size:', p)

if __name__ == '__main__':
    main()

