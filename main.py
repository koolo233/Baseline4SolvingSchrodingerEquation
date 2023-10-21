#参考文献：
#Physics-Informed Neural Networks_ A Deep Learning Framework for Solving Forward andInverse Problems Involving Nonlinear Partial Differential Equations
#
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, num_layers, num_neurons, input_dim, output_dim):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, num_neurons[0])
        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            layers.append(nn.Tanh())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_neurons[-1], output_dim)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = self.hidden_layers(out)
        out_final = self.output_layer(out)
        return out_final

#初始值采样
def create_data():
    x_lower = -5
    x_upper = 5
    t_lower = 0
    t_upper = np.pi/2

    # 初始点采样数据
    t_initial = np.zeros((100, 1))
    x_initial = np.random.uniform(low=x_lower, high=x_upper, size=(100, 1))
    t_initial_tensor = torch.from_numpy(t_initial).float().requires_grad_(True).to(device)
    x_initial_tensor = torch.from_numpy(x_initial).float().requires_grad_(True).to(device)
    initial_data = torch.cat([t_initial_tensor, x_initial_tensor],1)

    #边界点采样
    x_boundary_lower = -5 * np.ones((100, 1))
    x_boundary_upper = 5 * np.ones((100, 1))
    t = np.random.uniform(low=t_lower, high=t_upper, size=(100, 1))
    x_lower_tensor = torch.from_numpy(x_boundary_lower).float().requires_grad_(True).to(device)
    x_upper_tensor = torch.from_numpy(x_boundary_upper).float().requires_grad_(True).to(device)
    t_tensor = torch.from_numpy(t).float().requires_grad_(True).to(device)
    boundary_lower_data = torch.cat([t_tensor, x_lower_tensor],1)
    boundary_upper_data = torch.cat([t_tensor, x_upper_tensor],1)

    # 残差点采样数据
    x_collocation = np.random.uniform(low=x_lower, high=x_upper, size=(2000, 1))
    t_collocation = np.random.uniform(low=t_lower, high=t_upper, size=(2000, 1))
    x_collocation_tensor = torch.from_numpy(x_collocation).float().requires_grad_(True).to(device)
    t_collocation_tensor = torch.from_numpy(t_collocation).float().requires_grad_(True).to(device)
    pde_data = torch.cat([t_collocation_tensor, x_collocation_tensor],1)
    return pde_data, initial_data, boundary_lower_data, boundary_upper_data



def train(model, pde_data, initial_data, boundary_lower_data, boundary_upper_data):

    #定义导函数
    def gradients(output, input_tensor):
        gradients = torch.autograd.grad(outputs=output, inputs=input_tensor, grad_outputs=torch.ones_like(output),
                                        create_graph=True)[0]
        return gradients

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    losses = []
    initial_losses = []
    boundary_losses = []
    pde_losses = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # initial loss
        output_initial = model(initial_data)
        initial_value = 2 / torch.cosh(initial_data[:, 1:2])
        initial_loss_real = torch.mean((output_initial[:, 0:1] - initial_value) ** 2)
        initial_loss_imag = torch.mean((output_initial[:, 1:2]) ** 2)
        initial_loss = initial_loss_real + initial_loss_imag

        #boundary loss
        output_lower = model(boundary_lower_data)
        output_upper = model(boundary_upper_data)

        df_dx_lower_real = gradients(output_lower[:, 0:1], boundary_lower_data)[:, 1:2]
        df_dx_lower_imag = gradients(output_lower[:, 1:2], boundary_lower_data)[:, 1:2]

        df_dx_upper_real = gradients(output_upper[:, 0:1], boundary_upper_data)[:, 1:2]
        df_dx_upper_imag = gradients(output_upper[:, 1:2], boundary_upper_data)[:, 1:2]

        boundary_value_loss_real = torch.mean((output_lower[:, 0:1] - output_lower[:, 0:1]) ** 2)
        boundary_value_loss_imag = torch.mean((output_lower[:, 1:2] - output_lower[:, 1:2]) ** 2)

        boundary_gradient_loss_real = torch.mean((df_dx_lower_real - df_dx_upper_real) ** 2)
        boundary_gradient_loss_imag = torch.mean((df_dx_lower_imag - df_dx_upper_imag) ** 2)

        boundary_loss = boundary_value_loss_real + boundary_value_loss_imag + \
                              boundary_gradient_loss_real + boundary_gradient_loss_imag

        #PDE loss
        output = model(pde_data)
        output_real = output[:, 0:1]
        output_imag = output[:, 1:2]

        df_dtx_real = gradients(output_real, pde_data)
        df_dt_real = df_dtx_real[:, 0:1]
        df_dx_real = df_dtx_real[:, 1:2]

        df_dtx_imag = gradients(output_imag, pde_data)
        df_dt_imag = df_dtx_imag[:, 0:1]
        df_dx_imag = df_dtx_imag[:, 1:2]

        df_dxx_real = gradients(df_dx_real, pde_data)[:, 1:2]
        df_dxx_imag = gradients(df_dx_imag, pde_data)[:, 1:2]

        pde_real = -df_dt_imag + 0.5 * df_dxx_real + (output_real ** 2 + output_imag ** 2) * output_real
        pde_imag = df_dt_real + 0.5 * df_dxx_imag + (output_real ** 2 + output_imag ** 2) * output_imag
        pde_loss = torch.mean(pde_real**2)+torch.mean(pde_imag**2)

        # 将误差(损失)累加起来
        loss = initial_loss + boundary_loss + pde_loss

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        initial_losses.append(initial_loss.item())
        boundary_losses.append(boundary_loss.item())
        pde_losses.append(pde_loss.item())

        if epoch % 10 == 0:
            if not os.path.exists('log'):
                os.makedirs('log')
            with open('log/loss_PiNN.txt', 'a') as f:
                f.write(
                    'Epoch [{}/{}], Loss: {:.4f}, initial_loss: {:.8f}, boundary_loss: {:.8f}, pde_loss:{:.4f}\n'.format(
                        epoch + 1, epochs, loss.item(), initial_loss.item(), boundary_loss.item(),pde_loss.item()))

    # 绘制损失图像
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')
    plt.plot(range(1, len(pde_losses) + 1), pde_losses, label='pde_loss')
    plt.plot(range(1, len(boundary_losses) + 1), boundary_losses, label='boundary_loss')
    plt.plot(range(1, len(initial_losses) + 1), initial_losses, label='initial_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('log/loss_plot.png')
def pred_and_valuation(test):
    test_input_t = torch.from_numpy(test['input_t']).float().to(device)
    test_input_x = torch.from_numpy(test['input_x']).float().to(device)
    test_input_tensor = torch.stack([test_input_t, test_input_x], dim=1)
    test_output = test['output']
    with torch.no_grad():
        model.eval()
        pred_output = model(test_input_tensor.to(device)).detach().cpu().numpy()
        pred_length = np.sqrt(np.mean(pred_output[:, 0]**2)+np.mean(pred_output[:, 1]**2))
    L1 = np.linalg.norm(test_output - pred_length, 1)
    L2 = np.linalg.norm(test_output - pred_length)
    max_error = np.max(np.abs(pred_length-test_output))
    min_error = np.min(np.abs(pred_length-test_output))
    average_error = np.average(np.abs(pred_length-test_output))

    with open('log/pre_output.txt', 'a') as f:
        f.write(
            'l1: {:.8f}, l2: {:.8f}, max_error: {:.8f}, min_error:{:.8f}, average_error:{:.8f}\n'.format(
                L1, L2, max_error, min_error.item(), average_error))

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    pde_data, initial_data, boundary_lower_data, boundary_upper_data = create_data()
    model = PINN(num_layers=4, num_neurons=[100, 100, 100, 100], input_dim=2, output_dim=2).to(device)
    train(model, pde_data, initial_data, boundary_lower_data, boundary_upper_data)
    test = np.load('schrodingerEquation_GT.npz')
    pred_and_valuation(test)
