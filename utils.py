import torch
from torch import nn

class AdaptiveSILU(nn.Module):

    def __init__(self, scale_factor=1.0):
        super(AdaptiveSILU, self).__init__()

        self.param = nn.Parameter(torch.tensor(1.0))
        self.activate_func = nn.SiLU()
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.activate_func(self.param * x * self.scale_factor)


class AdaptiveTanh(nn.Module):

    def __init__(self, scale_factor=1.0):
        super(AdaptiveTanh, self).__init__()

        self.param = nn.Parameter(torch.tensor(1.0))
        self.activate_func = nn.Tanh()
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.activate_func(self.param * x * self.scale_factor)


def gpinn_loss(model, gradient_func, input_data, g_loss_w):

    output = model(input_data)  # PDE监督样本结果
    output_real = output[:, 0:1]  # 实部预测结果
    output_imag = output[:, 1:2]  # 虚部预测结果

    # 计算实部对输入的一阶梯度
    df_dtx_real = gradient_func(output_real, input_data)
    df_dt_real = df_dtx_real[:, 0:1]
    df_dx_real = df_dtx_real[:, 1:2]

    # 计算虚部对输入一阶梯度
    df_dtx_imag = gradient_func(output_imag, input_data)
    df_dt_imag = df_dtx_imag[:, 0:1]
    df_dx_imag = df_dtx_imag[:, 1:2]

    # 计算实部对输入坐标x的二阶梯度
    df_dxx_real = gradient_func(df_dx_real, input_data)[:, 1:2]
    # 计算虚部对输入坐标x的二阶梯度
    df_dxx_imag = gradient_func(df_dx_imag, input_data)[:, 1:2]

    # 计算实部PDE损失
    pde_real = -df_dt_imag + 0.5 * df_dxx_real + (output_real ** 2 + output_imag ** 2) * output_real
    # 计算虚部PDE损失
    pde_imag = df_dt_real + 0.5 * df_dxx_imag + (output_real ** 2 + output_imag ** 2) * output_imag
    # 计算总损失
    pde_loss = torch.mean(pde_real ** 2) + torch.mean(pde_imag ** 2)

    # 计算_dtx
    pde_real_dtx = gradient_func(pde_real, input_data)
    pde_imag_dtx = gradient_func(pde_imag, input_data)

    # 计算 gpinn loss
    pde_loss_ge = torch.mean(pde_real_dtx ** 2) + torch.mean(pde_imag_dtx ** 2)

    # 计算总损失
    loss = pde_loss + g_loss_w * pde_loss_ge
    return loss

