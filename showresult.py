import pandas as pd
import matplotlib.pyplot as plt
import os

# 构建结果目录路径
# train_dir = "E:/Data/public_training_data/training1/"
train_dir = "/homes/lwjiang/Data/IVIM/public_training_data/"
norm_train_dir1 = os.path.normpath(train_dir)
train_process_result = os.path.join(norm_train_dir1, "result/result.csv")

# Load the data
df = pd.read_csv(train_process_result)

# Plot
plt.figure(figsize=(20, 5))
# plot train or val loss
plt.subplot(1, 2, 1)
plt.plot(df["epoch"].values, df["train_loss_all"].values, label='Train Loss')
plt.plot(df["epoch"].values, df["val_loss_all"].values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

# plot train or val rMSE
plt.subplot(1, 2, 2)
plt.plot(df["epoch"].values, df["train_rRMSE_all"].values, label='Train rRMSE')
plt.plot(df["epoch"].values, df["val_rRMSE_all"].values, label='Validation rRMSE')
plt.xlabel('Epoch')
plt.ylabel('rRMSE')
plt.title('Training and Validation rRMSE')

plt.legend()
plt.show()
plt.savefig(os.path.join("/homes/lwjiang/Data/IVIM/public_training_data/result/","Loss&rRMSE.png"))

