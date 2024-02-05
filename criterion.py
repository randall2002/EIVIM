import torch
import numpy as np
from functions_and_demo import rRMSE_D, rRMSE_f, rRMSE_per_case, rRMSE_all_cases

# num_cases = 1000
# rRMSE_case =np.empty([num_cases])
# rRMSE_t_case =np.empty([num_cases])

def param_loss(model_out, gt_param, gt_t):
    rRMSE_case, rRMSE_t_case = rRMSE_per_case(model_out[0,:,:], model_out[1,:,:], model_out[2,:,:],\
                                                        gt_param[0,:,:], gt_param[1,:,:], gt_param[2,:,:], gt_t)
    loss = rRMSE_case + rRMSE_t_case
    return loss

# def img_loss(input, gt_img, num_cases):