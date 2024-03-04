import pandas as pd
import os


def generate_filename_with_identifier(base_filename, identifier, value):
#生成新的文件名。
    # Split the base filename to get name and extension
    name, extension = os.path.splitext(base_filename)
    # Format the value to ensure two decimal places for floats, replacing dot with underscore
    if isinstance(value, float):
        value_str = "{:0.2f}".format(value).replace('.', '_')
    else:
        value_str = str(value)
    # Construct the new filename with the identifier and its value
    new_filename = f"{name}_{identifier}_{value_str}{extension}"
    return new_filename

def save_alpha_info(alpha_info, train_dir):
    # 确保结果目录存在
    result_dir = os.path.join(train_dir, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 构建alpha信息的CSV文件路径
    alpha_info_path = os.path.join(result_dir, 'alpha_info.csv')

    # 将alpha信息转换为DataFrame并保存到CSV文件
    alpha_df = pd.DataFrame(alpha_info)
    #alpha_df.to_csv(alpha_info_path, index=False)
    #print(f"Alpha information saved to {alpha_info_path}")

    try:
        alpha_df.to_csv(alpha_info_path, index=False)
        print(f"Alpha information saved to {alpha_info_path}")
    except PermissionError:
        print(f"无法保存到 {alpha_info_path}。文件可能被其他程序占用。请关闭其他程序后再次尝试。")
        input("请关闭占用文件的程序后按Enter键继续...")
        try:
            alpha_df.to_csv(alpha_info_path, index=False)
            print(f"文件已成功保存至：{alpha_info_path}")
        except PermissionError:
            print(f"仍然无法保存到 {alpha_info_path}。请检查文件权限和占用情况。")


def test_generate_filename_with_identifier():

    # Example usage
    base_filename = 'U_net.pkl'
    identifier = 'alpha'
    value = 0.25

    # Generate the new filename
    new_filename = generate_filename_with_identifier(base_filename, identifier, value)
    print(new_filename)
    result_filename = generate_filename_with_identifier("result.csv", 'alpha', 0.1)
    print(result_filename)
    pass


def test_save_alpha_info():
    # 模拟alpha值及其对应的最佳损失
    alpha_info = [
        {'alpha': 0.0, 'loss': 0.002},
        {'alpha': 0.25, 'loss': 0.0015},
        {'alpha': 0.5, 'loss': 0.001},
        {'alpha': 0.75, 'loss': 0.0012},
        {'alpha': 1.0, 'loss': 0.4448}
    ]

    alpha_info.append({'best alpha': 0.75, 'best loss': 0.0012})
    # 指定存储alpha信息的目录
    train_dir = "E:/temp"  # 可以更改为您想要的目录

    # 调用函数保存alpha信息
    save_alpha_info(alpha_info, train_dir)
    pass

if __name__ == '__main__':
    test_generate_filename_with_identifier()
    test_save_alpha_info()

