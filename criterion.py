import torch
from IVIM_Dataset import read_data
from functions_and_demo import rRMSE_D, rRMSE_f, rRMSE_per_case, rRMSE_all_cases

def param_loss(input, gt_param, gt_t, num_cases):
    for i in range(num_cases):
        rRMSE_case[i], rRMSE_t_case[i] = rRMSE_per_case(input[:,:,0], input[:,:,1], input[:,:,2],\
                                                        gt_param[:,:,0], gt_param[:,:,1], gt_param[:,:,2], gt_t)
    rRMSE_final_1 = np.average(rRMSE_case)
    rRMSE_final_tumor_1 = np.average(rRMSE_t_case)

def img_loss(input, gt_img, )