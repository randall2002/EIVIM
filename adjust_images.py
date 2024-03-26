import numpy as np
import os
from tqdm import tqdm

class ImageStatistics:
    def __init__(self, gt_directory, result_directory):
        self.gt_directory = gt_directory
        self.result_directory = result_directory

    def read_data(self, file_dir, fname, i):

        file_dir = os.path.normpath(file_dir)
        fname_tmp = file_dir + "/{:04}".format(i) + fname
        data = np.load(fname_tmp)

        return data

    #要求：目录中所有的npy数据都是目标数据。
    def extract_numbers_from_filenames(self, directory, filter_name=None):

        numbers = []
        for filename in os.listdir(directory):
            if filename.endswith('.npy') and (filter_name is None or filter_name in filename):
                # 提取不带扩展名的部分，并转换为整数
                #number = int(filename.split('.')[0])
                number = int(filename[:4])
                numbers.append(number)
        return numbers

    def iterate_images(self, directory):
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                yield np.load(os.path.join(directory, filename))

    def compute_mean_std_per_image(self, directory, filter_name):
        means = []
        stds = []
        file_numbers = self.extract_numbers_from_filenames(directory, filter_name)
        for file_index in file_numbers:
            image = self.read_data(directory, filter_name, file_index)
            mean = np.mean(image, axis=(0, 1))  # 计算图像的均值，注意axis参数
            std = np.std(image, axis=(0, 1))  # 计算图像的标准差，注意axis参数
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)
        return means, stds

    def compute_global_mean_std(self, directory, filter_name):
        means, stds = self.compute_mean_std_per_image(directory, filter_name)
        global_mean = np.mean(means, axis=0)
        global_std = np.mean(stds, axis=0)
        return global_mean, global_std

    def adjust_and_save_images(self, alphas, target_directory):

        gt_mean, gt_std = self.compute_global_mean_std(self.gt_directory,'_IVIMParam.npy')

        for alpha in alphas:
            target_directory = os.path.normpath(target_directory)
            result_dir = f"{target_directory}_alpha_{alpha}"
            result_dir = result_dir.replace(".", "")
            os.makedirs(result_dir, exist_ok=True)
            file_numbers = self.extract_numbers_from_filenames(self.result_directory)

            for file_index in file_numbers:
                image = self.read_data(self.result_directory, '_IVIMParam.npy', file_index)
                self.adjust_zeromax_of_params(image)

                mean_i, std_i = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))

                mean_new = (mean_i - gt_mean) * alpha + gt_mean
                std_new = (std_i - gt_std) * alpha + gt_std
                adjusted_image = self.adjust_image(image, mean_new, std_new)

                save_file = '{}/{:04d}.npy'.format(result_dir, file_index)
                #os.path.join(result_dir, f"image_{i}.npy")
                np.save(save_file, adjusted_image)

    @staticmethod
    def adjust_image(image, mean_new, std_new):
        """
        根据新的均值和标准差调整图像。
        """
        mean_orig = np.mean(image, axis=(0, 1))
        std_orig = np.std(image, axis=(0, 1))
        adjusted_image = (image - mean_orig) / std_orig * std_new + mean_new
        return adjusted_image

    # 截断和置零处理。
    @staticmethod
    def adjust_zeromax_of_params(params):
        f_map, Dt_map, Ds_map = params[:, :, 0], params[:, :, 1], params[:, :, 2]

        # 1. Set negative values to zero for all parameters
        f_map[f_map < 0] = 0
        Dt_map[Dt_map < 0] = 0
        Ds_map[Ds_map < 0] = 0

        # 2. Truncate Dt values where Dt > 1/5 * max(Ds)
        max_Ds = np.max(Ds_map)
        Dt_threshold = max_Ds / 3
        Dt_map[Dt_map > Dt_threshold] = Dt_threshold

        # 3. Truncate Ds values where Ds > 20 * max(Dt)
        max_Dt = np.max(Dt_map)
        Ds_threshold = 30 * max_Dt
        Ds_map[Ds_map > Ds_threshold] = Ds_threshold

        # return f_map, Dt_map, Ds_map


if __name__ == "__main__":

    gt_params = "E:/IVIM-FIT-DATA/training1_gtparams"
    if False: #测试
        dir_result = "E:/IVIM-FIT-DATA/training2_result/fitdipy"
        dir_result_update = "E:/IVIM-FIT-DATA/training2_result/fitdipy_update"
    else: #validation:
        dir_result = "E:/IVIM-FIT-DATA/public_validation_data_result/jlw_pred"
        dir_result_update = "E:/IVIM-FIT-DATA/public_validation_data_result/jlw_pred_update"
    alphas = [0.9,0.8, 0.7, 0.6, 0.5, 0.4,0.3, 0.2,0.1]

    stats_a = ImageStatistics(gt_params, dir_result)

    stats_a.adjust_and_save_images( alphas, dir_result_update)
