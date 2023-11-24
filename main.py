"""
本文件为2023流光杯初赛Baseline。

本文件包含完整的数据创建、采样、模型构建、训练以及输出预测文件等功能，用于选手学习实践PINNs。

File Name: main.py
Author: Xiaomeng Wu
Created Date: 2023-10-
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN(nn.Module):
    """
    PINN 模型定义
    """
    def __init__(self, num_layers, num_neurons, input_dim=2, output_dim=2):
        """
        模型初始化
        :param num_layers: 总层数
        :param num_neurons: 每一层神经元数量
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        """
        super(PINN, self).__init__()

        # 输入层
        self.input_layer = nn.Linear(input_dim, num_neurons[0])

        # 隐藏层
        # 每一层由线性层和非线性激活函数构成
        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            layers.append(nn.Tanh())
        self.hidden_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Linear(num_neurons[-1], output_dim)

    def forward(self, x):
        """
        正向forward函数
        :param x: 输入数据
        :return: 结果
        """
        out = torch.tanh(self.input_layer(x))
        out = self.hidden_layers(out)
        out_final = self.output_layer(out)
        return out_final


def create_data():
    """
    创建用于监督PDE、Boundary以及Initial的数据集
    :return: 创建的数据集
    """

    # 根据待求解方程确定x范围和t范围
    x_lower = -5  # x最小值
    x_upper = 5   # x最大值
    t_lower = 0   # t最小值
    t_upper = np.pi/2   # t最大值

    # ----------
    # Initial
    # ----------
    # 创建Initial Condition采样点数据集
    # Initial Condition的t一定为0，x从[-5,5]随机采样
    # 总样本点数为100
    t_initial = np.zeros((100, 1))
    x_initial = np.random.uniform(low=x_lower, high=x_upper, size=(100, 1))
    # 生成tensor
    t_initial_tensor = torch.from_numpy(t_initial).float().to(device)
    x_initial_tensor = torch.from_numpy(x_initial).float().to(device)
    initial_data = torch.cat([t_initial_tensor, x_initial_tensor], dim=1)

    # ----------
    # Boundary
    # ----------
    # 创建Boundary Condition采样点数据集
    # 由于本次比赛的赛题为周期边界条件，因此需要在x的上下边界采样相同的x序列，t则在整个区间随机采样
    # 总样本点数为100
    x_boundary_lower = -5 * np.ones((100, 1))  # 下边界
    x_boundary_upper = 5 * np.ones((100, 1))  # 上边界
    t = np.random.uniform(low=t_lower, high=t_upper, size=(100, 1))  # 随机采样的t

    # 创建tensor
    x_lower_tensor = torch.from_numpy(x_boundary_lower).float().requires_grad_(True).to(device)
    x_upper_tensor = torch.from_numpy(x_boundary_upper).float().requires_grad_(True).to(device)
    t_tensor = torch.from_numpy(t).float().requires_grad_(True).to(device)
    boundary_lower_data = torch.cat([t_tensor, x_lower_tensor],1)
    boundary_upper_data = torch.cat([t_tensor, x_upper_tensor],1)

    # ----------
    # PDE
    # ----------
    # 创建PDE采样点数据集
    # 需要在区间内任意采样
    # 总样本点数为2000
    x_collocation = np.random.uniform(low=x_lower, high=x_upper, size=(2000, 1))
    t_collocation = np.random.uniform(low=t_lower, high=t_upper, size=(2000, 1))
    x_collocation_tensor = torch.from_numpy(x_collocation).float().requires_grad_(True).to(device)
    t_collocation_tensor = torch.from_numpy(t_collocation).float().requires_grad_(True).to(device)
    pde_data = torch.cat([t_collocation_tensor, x_collocation_tensor],1)

    # ----------
    # 绘制采样结果
    # ----------
    plt.figure(figsize=(10, 8))
    # 绘制Initial Condition采样点
    plt.scatter(x_initial, t_initial, c='r', marker='o', label='Initial')
    # 绘制Boundary Condition采样点
    plt.scatter(x_boundary_lower, t, c='b', marker='o', label='Boundary')
    plt.scatter(x_boundary_upper, t, c='b', marker='o')
    # 绘制PDE采样点
    plt.scatter(x_collocation, t_collocation, c='g', marker='o', label='PDE')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Collocation Points')
    plt.legend()
    plt.savefig('./log/Collocation Points.png')
    plt.clf()
    plt.close()

    return pde_data, initial_data, boundary_lower_data, boundary_upper_data


def train(model, pde_data, initial_data, boundary_lower_data, boundary_upper_data):
    """
    训练主函数
    本函数中将定义损失函数、优化器以及训练过程
    :param model: PINN模型
    :param pde_data: PDE数据
    :param initial_data: Initial数据
    :param boundary_lower_data: 下边界数据
    :param boundary_upper_data: 上边界数据
    :return: None
    """

    def gradients(_output, _input_tensor):
        """
        梯度计算
        :param _output: 输出tensor
        :param _input_tensor: 输入tensor
        :return: 输出对输入的梯度计算结果
        """
        _gradients = torch.autograd.grad(
            outputs=_output,
            inputs=_input_tensor,
            grad_outputs=torch.ones_like(_output),
            create_graph=True
        )[0]
        return _gradients

    losses = []  # 损失记录器
    initial_losses = []  # 初始条件损失记录器
    boundary_losses = []  # 边界条件损失记录器
    pde_losses = []  # PDE损失记录器

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 初始化优化器
    MAX_STEPS = 20  # 最大训练步数
    model.train()

    # 训练主循环
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()

        # -------------
        # 初始条件损失
        # -------------
        output_initial = model(initial_data)  # 预测
        initial_value = 2 / torch.cosh(initial_data[:, 1:2])  # 初始条件真实值
        initial_loss_real = torch.mean((output_initial[:, 0:1] - initial_value) ** 2)  # 实部
        initial_loss_imag = torch.mean((output_initial[:, 1:2]) ** 2)  # 虚部
        initial_loss = initial_loss_real + initial_loss_imag  # 初始条件损失

        # -------------
        # 边界条件损失
        # -------------
        output_lower = model(boundary_lower_data)  # 下边界预测结果
        output_upper = model(boundary_upper_data)  # 上边界预测结果

        # 下边界实部梯度
        df_dx_lower_real = gradients(output_lower[:, 0:1], boundary_lower_data)[:, 1:2]
        # 下边界虚部梯度
        df_dx_lower_imag = gradients(output_lower[:, 1:2], boundary_lower_data)[:, 1:2]

        # 上边界实部梯度
        df_dx_upper_real = gradients(output_upper[:, 0:1], boundary_upper_data)[:, 1:2]
        # 上边界虚部梯度
        df_dx_upper_imag = gradients(output_upper[:, 1:2], boundary_upper_data)[:, 1:2]

        # 周期边界条件：直接数值损失
        boundary_value_loss_real = torch.mean((output_lower[:, 0:1] - output_upper[:, 0:1]) ** 2)
        boundary_value_loss_imag = torch.mean((output_lower[:, 1:2] - output_upper[:, 1:2]) ** 2)

        # 周期边界条件：梯度损失
        boundary_gradient_loss_real = torch.mean((df_dx_lower_real - df_dx_upper_real) ** 2)
        boundary_gradient_loss_imag = torch.mean((df_dx_lower_imag - df_dx_upper_imag) ** 2)

        # 总损失
        boundary_loss = boundary_value_loss_real + boundary_value_loss_imag +\
                        boundary_gradient_loss_real + boundary_gradient_loss_imag

        # -------------
        # PDE损失
        # -------------
        output = model(pde_data)  # PDE监督样本结果
        output_real = output[:, 0:1]  # 实部预测结果
        output_imag = output[:, 1:2]  # 虚部预测结果

        # 计算实部对输入的一阶梯度
        df_dtx_real = gradients(output_real, pde_data)
        df_dt_real = df_dtx_real[:, 0:1]
        df_dx_real = df_dtx_real[:, 1:2]

        # 计算虚部对输入一阶梯度
        df_dtx_imag = gradients(output_imag, pde_data)
        df_dt_imag = df_dtx_imag[:, 0:1]
        df_dx_imag = df_dtx_imag[:, 1:2]

        # 计算实部对输入坐标x的二阶梯度
        df_dxx_real = gradients(df_dx_real, pde_data)[:, 1:2]
        # 计算虚部对输入坐标x的二阶梯度
        df_dxx_imag = gradients(df_dx_imag, pde_data)[:, 1:2]

        # 计算实部PDE损失
        pde_real = -df_dt_imag + 0.5 * df_dxx_real + (output_real ** 2 + output_imag ** 2) * output_real
        # 计算虚部PDE损失
        pde_imag = df_dt_real + 0.5 * df_dxx_imag + (output_real ** 2 + output_imag ** 2) * output_imag
        # 计算总损失
        pde_loss = torch.mean(pde_real**2)+torch.mean(pde_imag**2)

        # 误差累加
        loss = initial_loss + boundary_loss + pde_loss

        loss.backward()  # 反向传播
        optimizer.step()  # 参数优化

        # 记录损失值
        losses.append(loss.item())
        initial_losses.append(initial_loss.item())
        boundary_losses.append(boundary_loss.item())
        pde_losses.append(pde_loss.item())

        # 输出字符串
        print_str = 'Step [{}/{}], Loss: {:.4e}, initial_loss: {:.4e}, ' \
                    'boundary_loss: {:.4e}, pde_loss:{:.4e}'.format(
            MAX_STEPS, step, loss.item(), initial_loss.item(), boundary_loss.item(), pde_loss.item()
        )
        # 输出到log文件
        with open('log/loss_PINN.txt', 'a') as f:
            f.write(print_str + '\n')
        # 输出到cmd
        print(print_str)

    # 保存模型
    torch.save(model.state_dict(), "./log/model.pth")

    # 绘制损失图像
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')
    plt.plot(range(1, len(pde_losses) + 1), pde_losses, label='pde_loss')
    plt.plot(range(1, len(boundary_losses) + 1), boundary_losses, label='boundary_loss')
    plt.plot(range(1, len(initial_losses) + 1), initial_losses, label='initial_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('log/loss_plot.png')


def testing(test):

    # 创建test输入数据
    test_input_t = torch.from_numpy(np.array(test['t'])).float().to(device)
    test_input_x = torch.from_numpy(np.array(test['x'])).float().to(device)
    test_input_tensor = torch.stack([test_input_t, test_input_x], dim=1)

    # 模型推理
    model.eval()
    with torch.no_grad():
        pred = model(test_input_tensor.to(device)).detach()
        pred = torch.sqrt(pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2).cpu().numpy()
    pred = pred.reshape(-1)

    # 输出到csv
    df = pd.DataFrame()  # 创建DataFrame
    df["id"] = range(test['t'].shape[0])  # 创建id列
    df["t"] = test['t']  # 创建t列
    df["x"] = test['x']  # 创建x列
    df["pred"] = pred  # 创建pred列
    df.to_csv("log/baseline_submission.csv", index=False)


if __name__ == '__main__':
    # ----------
    # 固定随机数
    # ----------
    np.random.seed(0)  # numpy seed
    random.seed(0)  # random seed
    torch.manual_seed(0)  # pytorch seed
    torch.cuda.manual_seed(0)  # pytorch gpu seed

    # 创建日志文件夹
    if not os.path.exists('log'):
        os.makedirs('log')

    # ----------
    # 生成训练数据
    # ----------
    pde_data, initial_data, boundary_lower_data, boundary_upper_data = create_data()
    print("Init data done...")

    # ----------
    # 创建模型
    # ----------
    model = PINN(num_layers=4, num_neurons=[100, 100, 100, 100]).to(device)
    print("Init model done...")
    print(model)

    # ----------
    # 训练主循环
    # ----------
    train(model, pde_data, initial_data, boundary_lower_data, boundary_upper_data)
    print("Training done...")

    # ----------
    # 测试主循环
    # ----------
    testing(pd.read_csv('test.csv'))  # prediction
    print("Testing Done...")
    print("prediction file is saved as log/baseline_submission.csv")
