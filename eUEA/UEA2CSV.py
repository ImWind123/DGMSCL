from scipy.io import arff
import pandas as pd
import chardet
import torch
import numpy as np
import csv2arff

def csv_arff(csv_name, arff_name):
    try:
        csv2arff.csv2arff(csv_name, arff_name)
        print(f"成功将 {csv_name} 转换为 {arff_name}！")
    except Exception as e:
        print(f"转换失败：{e}")


def arff_csv(arff_name, csv_name):
    with open(arff_name, "rb") as f:
        data = f.read(4)  # 读取文件的前4个字节
        charset = chardet.detect(data)['encoding']  # 检测文件的编码方式
    with open(arff_name, "r", encoding=charset, errors="ignore") as f:
        data, meta = arff.loadarff(f)
        # 打印基本信息
        print(meta)
        df = pd.DataFrame(data)
        print(df.head())
        # 保存为csv文件
        output = pd.DataFrame(df)
        output.to_csv(csv_name, encoding='utf-8', index=False)


def arff_pt(arff_name, pt_name):
    with open(arff_name, "rb") as f:
        data = f.read(4)  # 读取文件的前4个字节
        charset = chardet.detect(data)['encoding']  # 检测文件的编码方式
    with open(arff_name, "r", encoding=charset, errors="ignore") as f:
        data, meta = arff.loadarff(f)
        # 获取属性和数据两部分
        # 打印基本信息
        print(meta)
        df = pd.DataFrame(data)
        df = df.iloc[:, :-1]
        # df = df.astype(np.float32)
        # 转换为tensor张量
        # df = df[:, :-1]
        tensor = torch.tensor(df.values)
        torch.save(tensor, pt_name)

# .arff文件中提取分类标签
def arff_pt_label(arff_name, pt_name):
    with open(arff_name, "rb") as f:
        data = f.read(4)  # 读取文件的前4个字节
        charset = chardet.detect(data)['encoding']  # 检测文件的编码方式
    with open(arff_name, "r", encoding=charset, errors="ignore") as f:
        data, meta = arff.loadarff(f)
        # 获取属性和数据两部分
        # 打印基本信息
        print(meta)
        df = pd.DataFrame(data)
        df = df.iloc[:, -1]
        df = df.astype(np.float32)
        tensor = torch.tensor(df.values)
        # tensor = torch.tensor(df.values)
        torch.save(tensor, pt_name)


# .csv文件中提取标签
def csv_pt_label(csv_name, pt_name):
    df = pd.read_csv(csv_name)
    label = df.iloc[:, -1]
    label = label.astype(np.float32)
    tensor = torch.tensor(df.values)
    torch.save(tensor, pt_name)
    # 获取属性部分
    # attributes = df.columns
    # 获取数据部分
    # data = df.iloc[1:]  # 假设第一行是属性行
    # # print("属性部分：")
    # # print(attributes)
    # # print("\n数据部分：")
    # # print(data)
    # # 获取数据列中的最后一列
    # label = data.iloc[:, -1]
    # df = label.astype(np.float32)
    # tensor = torch.tensor(df.values)
    # torch.save(tensor, pt_name)


def pt_csv(pt_name, csv_name):
    # 加载 .pt 文件中的数据
    data = torch.load(pt_name)
    # data = data.reshape(30, 640)
    # 将数据转换为 pandas 的 DataFrame 对象
    df = pd.DataFrame(data)

    # 将 DataFrame 对象保存为 .csv 文件
    df.to_csv(csv_name, index=False)

# csv_pt

def csv_pt(csv_name, pt_name):
    data = pd.read_csv(csv_name)
    # data = data.iloc[:, :-1]  #取除了最后一列以外的其它数据
    df = pd.DataFrame(data)
    df = df.astype(np.float32)
    tensor = torch.tensor(df.values, dtype=torch.float)
    torch.save(tensor, pt_name)

