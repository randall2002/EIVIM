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
def rRMSE_per_batch(model_out_batch, gt_param_batch, gt_t_batch):
    batch_size = model_out_batch.size(0)
    total_rRMSE = 0.0
    total_rRMSE_t = 0.0

    # 确保输入Tensor位于CPU上，并转换为NumPy数组
    model_out_batch_np = model_out_batch.cpu().numpy()
    gt_param_batch_np = gt_param_batch.cpu().numpy()
    gt_t_batch_np = gt_t_batch.cpu().numpy()

    for i in range(batch_size):
        model_out_case_np = model_out_batch_np[i]
        gt_param_case_np = gt_param_batch_np[i]
        gt_t_case_np = gt_t_batch_np[i]

        # 调用函数处理NumPy数组
        rRMSE_case, rRMSE_t_case = rRMSE_per_case(
            model_out_case_np[0], model_out_case_np[1], model_out_case_np[2],
            gt_param_case_np[0], gt_param_case_np[1], gt_param_case_np[2],
            gt_t_case_np
        )

        total_rRMSE += rRMSE_case
        total_rRMSE_t += rRMSE_t_case

    # 计算平均值
    avg_rRMSE = total_rRMSE / batch_size
    avg_rRMSE_t = total_rRMSE_t / batch_size

    return avg_rRMSE, avg_rRMSE_t

# def img_loss(input, gt_img, num_cases):