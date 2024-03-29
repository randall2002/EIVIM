import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import os
import zipfile
import numpy as np
from scipy.optimize import curve_fit

# 自定义转换，将 NumPy 数组转换为 PyTorch 张量，并调整通道顺序
class NumpyToTensor(object):
    def __call__(self, sample):
        # 调整通道顺序并转换为张量
        image = np.transpose(sample, (2, 0, 1))
        return torch.from_numpy(image)

class MyTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.fname_noisyDWIk = '_NoisyDWIk.npy'
        self.transform = transform

        self.start_index, self.count = self.__get_start_number_and_count(data_dir, self.fname_noisyDWIk)

        print(f"Start Number: {self.start_index}, Total Files: {self.count}")
 
    def __len__(self):
        return self.count

    def __getitem__(self, index):

        
        noisy_images, sk = self.load_noisy_images(index)

        # 应用数据增强
        #transform会导致维度顺序发生变化，所以改成一致维度。

        #维度顺序，变换前：（height, width, channels),如：(200, 200, 8)
        #维度顺序，变换后：（channels, height, width)，如：(8, 200, 200)

        if self.transform:
            angle = transforms.RandomRotation.get_params([-10, 10])  # 随机角度
            flip = torch.rand(1) < 0.5  # 随机决定是否翻转
            #应用变换
            noisy_images = self.apply_transform(noisy_images, angle, flip)
        else:
            numpy_to_tensor = NumpyToTensor()
            noisy_images = numpy_to_tensor(noisy_images)
    
        if not torch.is_tensor(noisy_images):
            raise TypeError("Returned data is not a tensor")

        # 返回样本和标签数据
        return noisy_images, index + self.start_index

    def load_noisy_images(self, index):
        # 加载带噪声的图像
        i = index + self.start_index
        #
        fname_without_ext, _ = os.path.splitext(self.fname_noisyDWIk)
        processed_fname =  os.path.join(self.data_dir, "{:04}".format(i) + fname_without_ext + "_processed")
        #refresh = True
        #if not refresh and os.path.exists(processed_fname):
        #不使用刷新机制，因为需要改写代码；使用删除数据机制：如果预处理代码改动，删除原来的预处理数据触发预处理流程。
        # if os.path.exists(processed_fname):
        #     # 如果预处理后的文件存在，则直接加载
        #     temp = np.load(processed_fname)
        #     noisy_preprocessed = temp['noisy_preprocessed']
        #     sk = temp['sk']

        # else:
            # 如果预处理后的文件不存在，则进行预处理
        noisy_k = self.__read_data(self.data_dir, self.fname_noisyDWIk, i)
        noisy_images = np.abs(np.fft.ifft2(noisy_k, axes=(0, 1), norm='ortho'))
        noisy_images = noisy_images.astype(np.float32)
        noisy_preprocessed, sk = self.__preprocess_1bseries(noisy_images)
            # 保存预处理后的数据和sk值
        np.save(processed_fname, noisy_preprocessed, sk)

        return noisy_preprocessed, sk

    def __preprocess_1bseries(self, noisy_images):



        # 计算首个b值图像的全局均值sk
        sk = np.mean(noisy_images[:, :, 0])
        # 对每个像素的信号除以sk
        noisy_images /= sk
        # -------------
        # 当使用这种归一化的预处理数据进行训练时，相应的无噪b系列图像也需要做同样的归一化。
        # --------------

        return noisy_images, sk



    def __get_start_number_and_count(self, data_dir, file_name):
        # 获取所有符合条件的文件名
        files = [f for f in os.listdir(data_dir) if file_name in f]

        # 如果没有找到文件，抛出异常或者返回默认值
        if not files:
            raise FileNotFoundError(f"No files with {file_name} found in {data_dir}")
        # 或者你可以返回默认值
        # return DEFAULT_START_NUMBER, DEFAULT_COUNT

        # 提取文件序号并转换为整数
        numbers = [int(f.split('_')[0]) for f in files]

        # 如果没有找到文件，返回None
        if not numbers:
            return None, None

        # 获取起始号和文件总数
        start_number = min(numbers)
        count = len(numbers)

        return start_number, count

    def __read_data(self, file_dir, fname, i):
        fname_tmp = file_dir + "{:04}".format(i) + fname
        data = np.load(fname_tmp)
        return data

    def apply_transform(self, image, angle, flip):

        # 如果图像是 NumPy 数组，先转换为 PyTorch 张量
        if isinstance(image, np.ndarray):
            numpy_to_tensor = NumpyToTensor()
            image = numpy_to_tensor(image)

        # 应用旋转和翻转变换
        transformed_image = TF.rotate(image, angle)
        if flip:
            transformed_image = TF.hflip(transformed_image)
        return transformed_image


# 测试代码；
import matplotlib.pyplot as plt


# 辅助函数，用于显示图像
def display_sample(noisy_images, noiseless_images, param_maps, tissue_images, sample_index, data_source):
    num_b_values = noisy_images.shape[0]
    fig, axs = plt.subplots(3, max(num_b_values, 3), figsize=(15, 6))

    for i in range(num_b_values):
        axs[0, i].imshow(noisy_images[i, :, :], cmap='gray')
        axs[0, i].set_title(f'Noisy b={i}')
        axs[0, i].axis('off')

        axs[1, i].imshow(noiseless_images[i, :, :], cmap='gray')
        axs[1, i].set_title(f'Noiseless b={i}')
        axs[1, i].axis('off')

    param_names = ['Param1', 'Param2', 'Param3']
    for i in range(3):
        axs[2, i].imshow(param_maps[i, :, :], cmap='gray')
        axs[2, i].set_title(param_names[i])
        axs[2, i].axis('off')

    axs[2, 3].imshow(tissue_images[0, :, :], cmap='gray')
    axs[2, 3].set_title(f'tissue')
    axs[2, 3].axis('off')

    plt.suptitle(f'Sample Index: {sample_index}, Data Source: {data_source}')
    plt.show()
    plt.savefig('case.png')


# 测试代码
def main():
    # data_dir = '/homes/lwjiang/Data/IVIM/public_training_data/training1/'
    data_dir = 'E:/Data/public_training_data/training2/'
    transform = transforms.Compose([
        NumpyToTensor(),  # 首先将Numpy数组转化为张量
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        # 可以添加更多的变换...
    ])

    dataset = MyDataset(data_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 加载一个批次的数据并显示一个样本
    for step, (noisy_images, (param_maps, noiseless_images, tissue_images), sample_indices) in enumerate(dataloader):
        if step == 10:  # 显示指定批次
            # 只显示第一个样本

            noisy_image = noisy_images[0].numpy()
            noiseless_image = noiseless_images[0].numpy()
            param_map = param_maps[0].numpy()
            tissue_image = tissue_images.numpy()
            sample_index = sample_indices[0]  # 获取第一个样本的索引

            # 显示样本
            print("noisy_image.shape: ", noisy_image.shape, "type: ", noisy_image.dtype)
            print("noiseless_image.shape: ", noiseless_image.shape, "type: ", noiseless_image.dtype)
            print("param_map.shape: ", param_map.shape, "type: ", param_map.dtype)
            display_sample(noisy_image, noiseless_image, param_map, tissue_image, sample_index, data_dir)
            break


if __name__ == '__main__':
    main()
# 实例化移到train.py
# Create training dataset and dataloader
# train_dataset = MyDataset(file_dir, fname_noisyDWIk, 1, num_train_cases)
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# Create validation dataset and dataloader
# val_dataset = MyDataset(file_dir, fname_noisyDWIk, num_train_cases + 1, num_train_cases + num_val_cases)
# val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
