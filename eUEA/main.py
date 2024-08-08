# 这是一个示例 Python 脚本。
import torch
import UEA2CSV
import pandas as pd
import numpy as np
import os
import dataset20

# arff2csv filename
arff_file = 'D:/GMSCLdataset/SelfRegulationSCP2/y_SRS2_test.arff'
csv_file = 'D:/GMSCLdataset/SelfRegulationSCP2/SelfRegulationSCP2Dimension6_TEST.csv'
# csv2pt filename
csv_name = 'D:/GMSCLdataset/SelfRegulationSCP2/SelfRegulationSCP2Dimension6_TEST.csv'
pt_name = 'D:/GMSCLdataset/SelfRegulationSCP2/SelfRegulationSCP2Dimension6_TEST.pt'
# traing files
X_train_name = 'D:/GMSCLdataset/MotorImagery/X_train.pt'
X_valid_name = 'D:/GMSCLdataset/MotorImagery/X_valid.pt'
Y_train_name = 'D:/GMSCLdataset/MotorImagery/y_train.pt'
Y_valid_name = 'D:/GMSCLdataset/MotorImagery/y_valid.pt'

def csv2pt(csv_name, pt_name):
    data = pd.read_csv(csv_name)
    data = data.iloc[:, :-1]  
    df = pd.DataFrame(data)
    df = df.astype(np.float32)
    tensor = torch.tensor(df.values, dtype=torch.float)
    torch.save(tensor, pt_name)

# 批量读取.arff文件并将其转换为.csv文件
def arff2csv_Batch(arff_folder):
    # 获取文件夹中的所有.arff文件
    arff_files = [f for f in os.listdir(arff_folder) if f.endswith('.arff')]
    # 遍历每个.arff文件
    for arff_file in arff_files:
        # 构建输入和输出文件路径
        input_path = os.path.join(arff_folder, arff_file)
        output_path = os.path.join(arff_folder, arff_file.replace('.arff', '.csv'))
        # 将.arff文件转换为.csv文件
        UEA2CSV.arff_csv(input_path, output_path)
        print(f"已将 {arff_file} 转换为 {output_path}")

# 批量转换.csv文件为.pt文件
def csv2pt_Batch(root_folder):
    # 批量转换.csv文件为.pt文件
    # 获取文件夹中的所有.csv文件
    csv_files = [f for f in os.listdir(root_folder) if f.endswith('.csv')]
    for csv_file in csv_files:
        # 构建输入和输出文件路径
        csv_path = os.path.join(root_folder, csv_file)
        pt_path = os.path.join(root_folder, csv_file.replace('.csv', '.pt'))
        # 将csv转换为pt
        csv2pt(csv_path, pt_path)
        print(f"已将 {csv_file} 转换为 {pt_path}")

#批量堆叠并转置X_train训练样本，得到X_train.pt
def X_train_pt_Batch(root_folder, X_train_name):
    # 批量读取训练集中的.pt文件
    pt_train_files = [f for f in os.listdir(root_folder) if f.endswith('_TRAIN.pt')]
    # 将每个train_pt文件torch.load
    X_train_list = []
    for i, pt_train_file in enumerate(pt_train_files):
        data = torch.load(os.path.join(root_folder, pt_train_file))
        data = torch.unsqueeze(data, dim=1)
        X_train_list.append(data)
        print(f"Loaded {pt_train_file}, shape: {data.shape}")
    # 将数据堆叠并转置
    X_train = torch.stack(X_train_list, dim=1)
    X_train = X_train.permute(0, 2, 1, 3)
    print(X_train.shape)
    torch.save(X_train, X_train_name)

#批量堆叠并转置X_valid训练样本，得到X_valid.pt
def X_valid_pt_Batch(root_folder, X_valid_name):
    # 批量读取训练集中的.pt文件
    pt_valid_files = [f for f in os.listdir(root_folder) if f.endswith('_TEST.pt')]
    # 将每个train_pt文件torch.load
    X_valid_list = []
    for i, pt_valid_file in enumerate(pt_valid_files):
        data = torch.load(os.path.join(root_folder, pt_valid_file))
        data = torch.unsqueeze(data, dim=1)
        X_valid_list.append(data)
        print(f"Loaded {pt_valid_file}, shape: {data.shape}")
    # 将数据堆叠并转置
    X_valid = torch.stack(X_valid_list, dim=1)
    X_valid = X_valid.permute(0, 2, 1, 3)
    print(X_valid.shape)
    torch.save(X_valid, X_valid_name)


if __name__ == '__main__':
    # 标签文件
    arff_train_file = 'D:/GMSCLdataset/5HMD/y_hmd_train.arff'
    arff_valid_file = 'D:/GMSCLdataset/5HMD/y_hmd_test.arff'
    # 训练文件
    X_train_name = 'D:/GMSCLdataset/5HMD/X_train.pt'
    X_valid_name = 'D:/GMSCLdataset/5HMD/X_valid.pt'
    Y_train_name = 'D:/GMSCLdataset/5HMD/y_train.pt'
    Y_valid_name = 'D:/GMSCLdataset/5HMD/y_valid.pt'
    # 批量读取.arff文件并将其转换为.csv文件
    # 设置文件夹路径
    root_folder = 'D:/GMSCLdataset/5HMD'
    # rate = 0.05
    # arff2csv_Batch(root_folder)  #批量将arff文件转换为csv文件
    # csv2pt_Batch(root_folder)   #批量将csv文件转换为pt文件
    # X_train_pt_Batch(root_folder, X_train_name)  #堆叠并转置训练样本，输出X_train.pt
    # X_valid_pt_Batch(root_folder, X_valid_name)  #堆叠并转置测试样本，输出X_valid.pt
    UEA2CSV.arff_pt_label(arff_train_file, Y_train_name)  #输出训练样本y_train.pt
    UEA2CSV.arff_pt_label(arff_valid_file, Y_valid_name)  # 输出测试样本y_valid.pt





