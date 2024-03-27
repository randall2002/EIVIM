import numpy as np
import matplotlib.pyplot as plt
from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table
from scipy.fftpack import ifftn
from scipy.optimize import curve_fit

import FittingMethod
#——————————————————————————————————————————————————————————————


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
#——————————————————————————————————————————————————————————————

if __name__ == '__main__':
    # file_dir = './data/train/'
    #file_Resultdir = './data/Result_dipy/'
    #file_Resultdir = './data/Result_dipy/'
    fitting_dir = 'E:/IVIM-FIT-DATA/training2/'
    fitting_resultdir = 'E:/IVIM-FIT-DATA/training2_result/'
    #file_resultdir = 'E:/IVIM-FIT-DATA/training2_result_DIPY/'

    gt_dir = fitting_dir
    fname_gt_param = '_IVIMParam.npy'
    fname_tissue = '_TissueType.npy'
    num_cases = 1000
    Nx = 200
    Ny = 200
    b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000])

    rRMSE_case = np.empty([num_cases])
    rRMSE_t_case = np.empty([num_cases])
    fitting = FittingMethod.Methods(fitting_dir, fitting_resultdir, b_values)
    # for i in range(num_cases):
    for i in range(980, 1000):
        # load gt data
        gt_param = FittingMethod.read_data(gt_dir, fname_gt_param, i + 1)

        # arr3D_fittedParams, save_file = fitting.method_gt_params(i+1)
        arr3D_fittedParams, save_file = fitting.method_fitdipy0(i + 1)

        # load tissue type data
        tissue = FittingMethod.read_data(gt_dir, fname_tissue, i + 1)

        # compute the rRMSE
        rRMSE_case[i], rRMSE_t_case[i] = rRMSE_per_case(arr3D_fittedParams[:, :, 0], arr3D_fittedParams[:, :, 1],
                                                        arr3D_fittedParams[:, :, 2], \
                                                        gt_param[:, :, 0], gt_param[:, :, 1], gt_param[:, :, 2], tissue)

        print('RMSE ALL {}\nRMSE tumor{}\nResult saved as {}'.format(rRMSE_case[i], rRMSE_t_case[i],
                                                                     save_file))

    # compute the average rRMSE for all cases
    rRMSE_final_1 = np.average(rRMSE_case)
    rRMSE_final_tumor_1 = np.average(rRMSE_t_case)
    print('Total RMSE all {}\n      RMSE tumor {}'.format(rRMSE_final_1, rRMSE_final_tumor_1))

    # save and zip data for submission
    # with zipfile.ZipFile('./Solution.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    #    for file in os.listdir(file_Resultdir):
    #        zipf.write(file_Resultdir+file,arcname=file)
