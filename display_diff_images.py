import numpy as np
import matplotlib.pyplot as plt


def display_group(img1, img2, group_index, axs, scale_diff=10, param_name=""):
    """
    Displays a pair of images and their difference on a specified row of subplots.

    Parameters:
    - img1, img2: Two numpy arrays representing images to compare.
    - row_index: Row index in the figure to display the images.
    - axs: Array of Axes objects in the figure.
    - scale_diff: Scale factor for the difference image.
    """
    #对两幅图像整体归一化，归一化目标[0,1]

    title_fontsize = 8
    max_val = np.max([img1.max(), img2.max()])
    img1_norm = img1 / max_val
    img2_norm = img2 / max_val
    diff_img = np.abs(img1_norm - img2_norm) * scale_diff

    axs[group_index, 0].imshow(img1_norm, cmap='gray', vmin=0, vmax=1)
    axs[group_index, 0].set_title(f"{param_name} Pred", fontsize=title_fontsize)
    axs[group_index, 0].axis('off')

    axs[group_index, 1].imshow(img2_norm, cmap='gray', vmin=0, vmax=1)
    axs[group_index, 1].set_title(f"{param_name} GT", fontsize=title_fontsize)
    axs[group_index, 1].axis('off')

    axs[group_index, 2].imshow(diff_img, cmap='gray', vmin=0, vmax=1)
    axs[group_index, 2].set_title(f"{param_name} Diffx{scale_diff}", fontsize=title_fontsize)
    axs[group_index, 2].axis('off')

def display_images(img_pairs, param_names, scale_diff=10):
    num_rows = len(img_pairs)
    fig, axs = plt.subplots(num_rows, 3, figsize=(10, 2 * num_rows))

    for i, (img1, img2) in enumerate(img_pairs):
        display_group(img1, img2, i, axs, scale_diff, param_name=param_names[i])

    plt.tight_layout()
    plt.show()

#显示参数图预测和真实之差
def show_params_diff():

    pred_params = np.load('D:/PYTHON/PyTorch/others/IVIM-test/Data/Result/0001.npy')
    gt_params = np.load('D:/PYTHON/PyTorch/others/IVIM-test/Data/train/0001_IVIMParam.npy')

    # 将每个参数图像分离出来
    pred_f, pred_Dt, pred_Ds = [pred_params[:, :, i] for i in range(3)]
    gt_f, gt_Dt, gt_Ds = [gt_params[:, :, i] for i in range(3)]

    img_pairs = [(pred_f, gt_f), (pred_Dt, gt_Dt), (pred_Ds, gt_Ds)]  # Example with two pairs

    param_names = ['f', 'Dt', 'Ds']  # Parameter names

    display_images(img_pairs, param_names)


#显示图像差：
def show_dwis_diff():
    gt_images = np.abs(np.load('D:/PYTHON/PyTorch/others/IVIM-test/Data/Train/0001_gtDWIs.npy'))

    noisy_k = np.load('D:/PYTHON/PyTorch/others/IVIM-test/Data/Train/0001_NoisyDWIk.npy')
    noisy_images = np.abs(np.fft.ifft2(noisy_k, axes=(0, 1), norm='ortho'))
    b_values = np.array([0, 5, 50, 100, 200])#, 500, 800, 1000])
    #b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000])

    # 准备图像对和它们的参数名
    img_pairs = []
    param_names = []
    for index, b_value in enumerate(b_values):
        gt_img = gt_images[:, :, index]
        noisy_img = noisy_images[:, :, index]
        img_pairs.append((noisy_img, gt_img))
        param_names.append(f'b={b_value}')

    # 使用display_images来展示每个b值对应的图像对
    display_images(img_pairs, param_names)

    pass

#测试一行显示一组对比图像
def test_main():
    # Example usage:
    img1 = np.random.rand(100, 100)
    img2 = img1 * 1.1  # Slightly altered version of img1
    img_pairs = [(img1, img2), (img2, img1 * 1.2), (img2, img1 * 1.3) ]  # Example with two pairs

    display_images(img_pairs)
# Example usage
if __name__ == "__main__":
    #test_main()
    #show_params_diff()
    show_dwis_diff()
