import torch.nn.functional as F

class PatchDataset(Dataset):
    def __init__(self, data_dir, transform=None, patch_size=(25, 25)):
        self.data_dir = data_dir
        self.fname_gt = '_IVIMParam.npy'
        self.fname_tissue = '_TissueType.npy'
        self.fname_noisyDWIk = '_NoisyDWIk.npy'
        self.fname_gtDWIs = '_gtDWIs.npy'
        self.transform = transform
        self.patch_size = patch_size

        self.start_index, self.count = self.__get_start_number_and_count(data_dir, self.fname_noisyDWIk)

        print(f"Start Number: {self.start_index}, Total Files: {self.count}")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        tissue_image = self.load_tissue_image(index)
        noisy_images, sk = self.load_noisy_images(index, tissue_image)
        param_maps = self.load_param_maps(index)
        noiseless_images = self.load_noiseless_images(index) / sk

        # 预处理后，将带噪图像切割成多个 patch
        patches = self.generate_patches(noisy_images, self.patch_size)

        # 将每个 patch 送入全连接网络
        outputs = []
        for patch in patches:
            patch_tensor = torch.from_numpy(patch)
            output = your_fully_connected_network(patch_tensor)  # 使用你自己的全连接网络
            outputs.append(output)

        # 返回样本和标签数据
        return outputs, (param_maps, noiseless_images, tissue_image), index + self.start_index

    def generate_patches(self, noisy_images, patch_size):
        patches = []
        for i in range(0, noisy_images.shape[1] - patch_size[0] + 1, patch_size[0]):
            for j in range(0, noisy_images.shape[2] - patch_size[1] + 1, patch_size[1]):
                patch = noisy_images[:, i:i+patch_size[0], j:j+patch_size[1]]
                patches.append(patch)
        return patches

    # 其他代码不变

    import torch
from monai.transforms import PatchExtraction

# 假设你的数据是一个大小为(1, 200, 200)的三维张量
image = torch.randn(1, 200, 200)

# 定义patch的大小
patch_size = (8, 8)

# 定义patch的数量
num_patches = (25, 25)

# 使用PatchExtraction进行patch裁剪
patch_extraction = PatchExtraction(patch_size, num_patches)
patches = patch_extraction(image)

# 打印裁剪后的patch数量
print("Number of patches:", len(patches))

# 打印裁剪后的第一个patch的大小
print("Size of the first patch:", patches[0].shape)

