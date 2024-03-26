import numpy as np
import os
from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table
from scipy.fftpack import ifftn
from scipy.optimize import curve_fit

# this function reads an npy file for case i
# need to use 'import  numpy as np'
def read_data(file_dir, fname, i):
    fname_tmp = file_dir + "{:04}".format(i) + fname
    data = np.load(fname_tmp)

    return data
def fit_biExponential_model_dipy(arr3D_imgk, arr1D_b):
    # 转换k空间数据到实空间
    arr3D_img = np.abs(np.fft.ifft2(arr3D_imgk, axes=(0, 1), norm='ortho'))

    bvecs = np.zeros((len(arr1D_b), 3))  # 假设的b向量，这里我们用零向量代替
    bvecs[1:, ] = np.array([1, 0, 0])
    # 创建梯度表
    gtab = gradient_table(arr1D_b, bvecs,b0_threshold=0)

    # 初始化IVIM模型
    ivim_model = IvimModel(gtab, fit_method='trr')#fit_method='lsq'，fit_method='trr' fit_method=’VarPro’

    # 预分配参数图数组
    arr2D_fFitted = np.zeros((arr3D_img.shape[0], arr3D_img.shape[1]))
    arr2D_DtFitted = np.zeros((arr3D_img.shape[0], arr3D_img.shape[1]))
    arr2D_DsFitted = np.zeros((arr3D_img.shape[0], arr3D_img.shape[1]))

    # 遍历体素
    for i in range(arr3D_img.shape[0]):
        for j in range(arr3D_img.shape[1]):
            # 仅对体素内部进行拟合
            if arr3D_img[i, j, 0] > 0:
                # 拟合IVIM模型
                ivim_fit = ivim_model.fit(arr3D_img[i, j])

                # 提取拟合参数
                arr2D_fFitted[i, j] = ivim_fit.perfusion_fraction
                arr2D_DtFitted[i, j] = ivim_fit.D
                arr2D_DsFitted[i, j] = ivim_fit.D_star

    # 组合参数图
    return np.stack((arr2D_fFitted, arr2D_DtFitted, arr2D_DsFitted), axis=-1)


# this function performs a basic reconstruction (inverse FT, then pixel-wise data fitting on magnitude image).
# need to use 'from scipy.optimize import curve_fit'

# bi-exponential function
def funcBiExp(b, f, Dt, Ds):
    ## Units
    # b: s/mm^2
    # D: mm^2/s
    return (1. - f) * np.exp(-1. * Dt * b) + f * np.exp(-1. * Ds * b)


def fit_biExponential_model(arr3D_imgk, arr1D_b):
    arr3D_img = np.abs(np.fft.ifft2(arr3D_imgk, axes=(0, 1), norm='ortho'))

    arr2D_coordBody = np.argwhere(arr3D_img[:, :, 0] > 0)
    arr2D_fFitted = np.zeros_like(arr3D_img[:, :, 0])
    arr2D_DtFitted = np.zeros_like(arr3D_img[:, :, 0])
    arr2D_DsFitted = np.zeros_like(arr3D_img[:, :, 0])

    for arr1D_coord in arr2D_coordBody:
        try:
            popt, pcov = curve_fit(funcBiExp, arr1D_b[1:] - arr1D_b[0],
                                   arr3D_img[arr1D_coord[0], arr1D_coord[1], 1:] / arr3D_img[
                                       arr1D_coord[0], arr1D_coord[1], 0]
                                   , p0=(0.15, 1.5e-3, 8e-3), bounds=([0, 0, 3.0e-3], [1, 2.9e-3, np.inf]),
                                   method='trf')
        except:
            popt = [0, 0, 0]
            print('Coord {} fail to be fitted, set all parameters as 0'.format(arr1D_coord))

        arr2D_fFitted[arr1D_coord[0], arr1D_coord[1]] = popt[0]
        arr2D_DtFitted[arr1D_coord[0], arr1D_coord[1]] = popt[1]
        arr2D_DsFitted[arr1D_coord[0], arr1D_coord[1]] = popt[2]

    return np.concatenate(
        (arr2D_fFitted[:, :, np.newaxis], arr2D_DtFitted[:, :, np.newaxis], arr2D_DsFitted[:, :, np.newaxis]), axis=2)


class Methods:
    def __init__(self, fitting_dir, fitting_resultdir_base, b_values):
        self.file_dir = fitting_dir
        self.fitting_resultdir_base = fitting_resultdir_base
        #fname_tissue = '_TissueType.npy'
        #fname_gt_image = '_gtDWIs.npy'
        self.fname_noisyDWIk = '_NoisyDWIk.npy'
        self.b_values = b_values

    def method_fit0(self, result_index):
        # load noisy data and perform baseline reconstruction
        noisy_DWIk = read_data(self.file_dir, self.fname_noisyDWIk, result_index)

        arr3D_fittedParams = fit_biExponential_model(noisy_DWIk, self.b_values)
        result_dir = f"{self.fitting_resultdir_base}fit0"
        os.makedirs(result_dir, exist_ok=True)
        save_dir = self.save_result(arr3D_fittedParams, result_dir, result_index)
        return arr3D_fittedParams, save_dir


    def method_fitdipy(self, result_index):
        # load noisy data and perform baseline reconstruction
        noisy_DWIk = read_data(self.file_dir, self.fname_noisyDWIk, result_index)

        arr3D_fittedParams = fit_biExponential_model_dipy(noisy_DWIk, self.b_values)

        result_dir = f"{self.fitting_resultdir_base}fitdipy"
        os.makedirs(result_dir, exist_ok=True)
        save_file = self.save_result(arr3D_fittedParams, result_dir, result_index)
        return arr3D_fittedParams, save_file
    def method_fitdipy0(self, result_index):
        # load noisy data and perform baseline reconstruction
        noisy_DWIk = read_data(self.file_dir, self.fname_noisyDWIk, result_index)

        arr3D_fittedParams = fit_biExponential_model_dipy(noisy_DWIk, self.b_values)

        self.adjust_diffusion_params(arr3D_fittedParams)

        result_dir = f"{self.fitting_resultdir_base}fitdipy0"
        os.makedirs(result_dir, exist_ok=True)
        save_file = self.save_result(arr3D_fittedParams, result_dir, result_index)
        return arr3D_fittedParams, save_file
    def save_result(self, fitting_result, result_dir, result_index):
        save_file = '{}/{:04d}.npy'.format(result_dir, result_index)
        np.save(save_file, fitting_result)
        return save_file


    #截断和置零处理。
    @staticmethod
    def adjust_diffusion_params(params):

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

        #return f_map, Dt_map, Ds_map

