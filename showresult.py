import pandas as pd
import matplotlib.pyplot as plt
import os

# 构建结果目录路径
train_dir = "E:/Data/public_training_data/training1/"
norm_train_dir1 = os.path.normpath(train_dir)
train_process_result = os.path.join(os.path.dirname(norm_train_dir1), "result/result.csv")

# Load the data
df = pd.read_csv(train_process_result)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train_loss_all"], label='Train Loss')
plt.plot(df["epoch"], df["val_loss_all"], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
