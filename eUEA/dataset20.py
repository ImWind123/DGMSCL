import pandas as pd
import random
import os

def csvTrans(root_folder, rate):
    # 获取文件夹中所有文件
    file_list = os.listdir(root_folder)
    # 遍历每个文件
    for file_name in file_list:
        if file_name.endswith('_TRAIN.csv') or file_name.endswith('_TEST.csv') or file_name.endswith('_train.csv') or file_name.endswith('_test.csv'):
            file_path = os.path.join(root_folder, file_name)
            df = pd.read_csv(file_path)
            left_count = df[df.iloc[:, -1] == "b'forward'"].shape[0]
            # 计算要保留的行数
            keep_count = int(left_count * rate)
            # 随机选择要删除的行
            rows_to_delete = random.sample(range(left_count), left_count - keep_count)
            # 删除标签为“left”的数据
            df.drop(rows_to_delete, inplace=True)
            # 保存修改后的文件
            df.to_csv(file_path, index=False)
            # 保存一次，打印一次
            print(f'保存{file_path},成功')