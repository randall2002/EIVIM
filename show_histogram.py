import numpy as np
import matplotlib.pyplot as plt


def compute_mean_std(data):
    """
    Compute the mean and standard deviation of t he given data.
    """
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def display_histogram(data, bins=10, hist_range=None, data_source="Data Source Not Provided", rwidth=0.8):
    """
    显示给定数据的直方图。
    """
    data = data.ravel()
    if hist_range is None:
        plt.hist(data, bins=bins, color='skyblue', edgecolor='black', rwidth=rwidth)
    else:
        plt.hist(data, bins=bins, range=hist_range, color='skyblue', edgecolor='black', rwidth=rwidth)

        # 计算数据属性
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_val = np.std(data)

    # 添加数据属性文本
    data_info = f"Min: {min_val:.5f}\nMax: {max_val:.5f}\nMean: {mean_val:.5f}\nStd: {std_val:.5f}"
    plt.gca().text(0.95, 0.95, data_info, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"Histogram - {data_source}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def stat_histogram(data, bins=10, hist_range=None):
    """
    统计给定数据的直方图并返回统计信息。
    """
    if hist_range is None:
        hist, bin_edges = np.histogram(data, bins=bins)
    else:
        hist, bin_edges = np.histogram(data, bins=bins, range=hist_range)
    return hist, bin_edges

#显示扩散参数图每个参数的直方图分布
def test_functions():
    #import matplotlib
    #matplotlib.use('TkAgg')  # 也可以尝试使用 'Qt5Agg', 'GTK3Agg', 等。

    # 生成测试数据
    # data = np.random.normal(loc=0, scale=1, size=10000)
    # data[0] = -30.0
    # data_source ='E:/IVIM-FIT-DATA/training2_result/0981.npy'
    data_source = 'E:/IVIM-FIT-DATA/training2_result_DIPY/0981.npy'

    data = np.load(data_source)

    for i in range(3):
        datai = data[:,:, i]
        # 测试计算平均值和标准差的函数
        mean, std = compute_mean_std(datai)
        print(f"Mean: {mean}, Std: {std}")

        # hist_range2=None #(0,datai.max())
        hist_range2 = (0.02*datai.max(),datai.max())

        # 测试显示直方图的函数
        datai_source = f"{data_source.split('/')[-2]}/{data_source.split('/')[-1]}: {i} "
        display_histogram(datai, bins=20, hist_range=hist_range2, data_source=datai_source)

        # 测试统计直方图的函数
        hist, bin_edges = stat_histogram(datai, bins=20, hist_range=hist_range2)
        print("Histogram bins:", bin_edges)
        print("Histogram counts:", hist)


if __name__ == "__main__":
    test_functions()
