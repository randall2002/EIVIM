# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:57:52 2023

@author: Xiaoyu Hu
"""
import os
import zipfile
import numpy as np
from scipy.optimize import curve_fit


# this function unzips the target file to the destination path.
# need to use 'import zipfile'
def unzip_data(target_dir, des_dir):
    zipfile.ZipFile(target_dir, 'r').extractall(des_dir)


# this function reads an npy file for case i
# need to use 'import  numpy as np'
def read_data(file_dir, fname, i):
    fname_tmp = file_dir + "{:04}".format(i) + fname
    data = np.load(fname_tmp)

    return data


# this fucntion computes the rRMSE for one microstructual parameter image for each case
# y is the reference solution
# t is the tissue type image
def rRMSE_D(x, y, t):
    Nx = x.shape[0]
    Ny = x.shape[1]

    t_tmp = np.reshape(t, (Nx * Ny,))
    tumor_indice = np.argwhere(t_tmp == 8)
    non_tumor_indice = np.argwhere(t_tmp != 8)
    non_air_indice = np.argwhere(t_tmp != 1)
    non_tumor_air_indice = np.intersect1d(non_tumor_indice, non_air_indice)

    x_tmp = np.reshape(x, (Nx * Ny,))
    x_t = x_tmp[tumor_indice]
    x_nt = x_tmp[non_tumor_air_indice]

    y_tmp = np.reshape(y, (Nx * Ny,))
    y_t = y_tmp[tumor_indice]
    y_nt = y_tmp[non_tumor_air_indice]

    # tumor region
    tmp1 = np.sqrt(np.sum(np.square(y_t)))
    tmp2 = np.sqrt(np.sum(np.square(x_t - y_t)))
    z_t = tmp2 / tmp1

    # non-tumor region
    tmp1 = np.sqrt(np.sum(np.square(y_nt)))
    tmp2 = np.sqrt(np.sum(np.square(x_nt - y_nt)))
    z_nt = tmp2 / tmp1

    return z_t, z_nt


def rRMSE_f(x, y, t):
    Nx = x.shape[0]
    Ny = x.shape[1]

    t_tmp = np.reshape(t, (Nx * Ny,))
    tumor_indice = np.argwhere(t_tmp == 8)
    non_tumor_indice = np.argwhere(t_tmp != 8)
    non_air_indice = np.argwhere(t_tmp != 1)
    non_tumor_air_indice = np.intersect1d(non_tumor_indice, non_air_indice)

    x_tmp = np.reshape(x, (Nx * Ny,))
    x_t = x_tmp[tumor_indice]
    x_nt = x_tmp[non_tumor_air_indice]

    y_tmp = np.reshape(y, (Nx * Ny,))
    y_t = y_tmp[tumor_indice]
    y_nt = y_tmp[non_tumor_air_indice]

    # tumor region
    tmp1 = np.sqrt(tumor_indice.shape[0])
    tmp2 = np.sqrt(np.sum(np.square(x_t - y_t)))
    z_t = tmp2 / tmp1

    # non-tumor region
    tmp1 = np.sqrt(non_tumor_air_indice.shape[0])
    tmp2 = np.sqrt(np.sum(np.square(x_nt - y_nt)))
    z_nt = tmp2 / tmp1

    return z_t, z_nt


# this fucntion computes the rRMSE for one case
def rRMSE_per_case(x_f, x_dt, x_ds, y_f, y_dt, y_ds, t):
    R_f_t, R_f_nt = rRMSE_f(x_f, y_f, t)
    R_Dt_t, R_Dt_nt = rRMSE_D(x_dt, y_dt, t)
    R_Ds_t, R_Ds_nt = rRMSE_D(x_ds, y_ds, t)

    z = (R_f_t + R_Dt_t + R_Ds_t) / 3 + (R_f_nt + R_Dt_nt) / 2

    z_t = (R_f_t + R_Dt_t + R_Ds_t) / 3

    return z, z_t


# this fucntion computes the rRMSE for all cases
# y is the reference solution
def rRMSE_all_cases(x_f, x_dt, x_ds, y_f, y_dt, y_ds, t):
    z = np.empty([x_f.shape[2]])
    z_t = np.empty([x_f.shape[2]])

    for i in range(x_f.shape[2]):
        z[i], z_t[i] = rRMSE_per_case(x_f[:, :, i], x_dt[:, :, i], x_ds[:, :, i], y_f[:, :, i], y_dt[:, :, i],
                                      y_ds[:, :, i], t[:, :, i])

    return np.average(z), np.average(z_t)

import os

def extract_numbers_from_filenames(directory):
    numbers = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            # 提取不带扩展名的部分，并转换为整数
            number = int(filename.split('.')[0])
            numbers.append(number)
    return numbers


# 在这个函数中，我们将把打印信息保存到一个文本文件中
def save_print_info(directory, rrmse_cases, rrmse_tumor_cases):
    # 创建文件路径
    output_file_path = os.path.join(directory, 'evaluation_summary.txt')

    # 使用with语句来确保文件在写入后正确关闭
    with open(output_file_path, 'w') as file:
        file.write(f'evaluation result:{directory}\n')
        file.write('RMSE Evaluation Summary\n')
        file.write('----------------------\n')
        file.write(f'Average of Total RMSE: {np.mean([case[1] for case in rrmse_cases])}\n')
        file.write(f'Average of Total RMSE_tumor: {np.mean([case[1] for case in rrmse_tumor_cases])}\n')
        file.write('Case by case RMSE:\n')

        # 遍历所有案例，打印每个案例的RMSE
        for case in rrmse_cases:
            file_index, rrmse = case
            rrmse_tumor = next((tumor_case[1] for tumor_case in rrmse_tumor_cases if tumor_case[0] == file_index), None)
            file.write(f'Case {file_index}: RMSE ALL {rrmse}, RMSE tumor {rrmse_tumor}\n')

#对指定路径的结果（输出参数图文件列表）进行评价，并保存到当前目录中。
if __name__ == '__main__':
    # file_dir = './data/train/'
    # file_Resultdir = './data/Result_dipy/'
    # file_Resultdir = './data/Result_dipy/'
    gt_dir = 'E:/IVIM-FIT-DATA/training2/'
    result_dir = 'E:/IVIM-FIT-DATA/training2_result/fitdipy0_update_alpha_07/'
    # file_Resultdir = 'E:/IVIM-FIT-DATA/training2_result_DIPY/'
    number_list = extract_numbers_from_filenames(result_dir)

    #######################
    fname_gt_params = '_IVIMParam.npy'
    fname_tissue = '_TissueType.npy'
    #fname_gtimage = '_gtDWIs.npy'
    #fname_noisyDWIk = '_NoisyDWIk.npy'
    fname_result = '.npy'
    Nx = 200
    Ny = 200
    b = np.array([0, 5, 50, 100, 200, 500, 800, 1000])

    rRMSE_cases = []
    rRMSE_t_cases = []

    for file_index in number_list:
        # load gt data
        gt_param = read_data(gt_dir, fname_gt_params, file_index)

        # load tissue type data
        tissue = read_data(gt_dir, fname_tissue, file_index)

        # load fitted parameter map
        arr3D_fittedParams = read_data(result_dir, fname_result, file_index)

        # compute the rRMSE
        rrmse, rrmse_t_case = rRMSE_per_case(arr3D_fittedParams[:, :, 0], arr3D_fittedParams[:, :, 1],
                                                        arr3D_fittedParams[:, :, 2], \
                                                        gt_param[:, :, 0], gt_param[:, :, 1], gt_param[:, :, 2], tissue)
        rRMSE_cases.append((file_index, rrmse))
        rRMSE_t_cases.append((file_index, rrmse_t_case))

        print('RMSE ALL {}\nRMSE tumor{}\nEvaluation file: {}'.format(rrmse, rrmse_t_case,
                                                                     '{}/{:04d}.npy'.format(file_Resultdir, file_index)))
    # compute the average rRMSE for all cases
    rRMSE_final_1 = np.mean([case[1] for case in rRMSE_cases])
    rRMSE_final_tumor_1 = np.mean([case[1] for case in rRMSE_t_cases])
    print('Total RMSE all {}\n      RMSE tumor {}'.format(rRMSE_final_1, rRMSE_final_tumor_1))

    save_print_info( result_dir, rRMSE_cases, rRMSE_t_cases )
