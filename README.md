# PINNs for Solving Schrodinger Equation

## Archive this repo

本项目不再更新，所有数据均已整理到[Kaggle](https://www.kaggle.com/datasets/zijiangyang1116/solve-pdes-with-ai-schrdinger-equation/data)。

## 运行环境

首先需要使用conda配置环境，本Baseline基于Python 3.8，PyTorch1.13.0。初赛在个人电脑上，显存大于2G就能正常训练以及测试。
下述命令如果出现类似于`http erro`或是`retry XXX`，请切换为国内源重试。

```commandline
conda create -n pinn python=3.8
conda activate pinn
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

克隆本Baseline:
```commandline
git clone git@github.com:koolo233/Baseline4SolvingSchrodingerEquation.git
cd Baseline4SolvingSchrodingerEquation
```

## 本Baseline简介

本Baseline以基础PINN为模型，基于PyTorch实现了初赛的Baseline。包含从数据采样、模型构建、训练、测试以及生成提交文件的完整流程。大家可以在本Baseline的基础上进行修改，实现自己的想法。亦或是使用本Baseline的代码作为参考，实现自己的框架代码。

对于PINN基础理论以及算法框架在此不多赘述，对这部分还有疑问的选手可以参考一并附带的参考文献以及PPT。

本Baseline的代码注释比较详细，对于代码中的一些细节可以参考注释。

各文件介绍如下：
1. `main.py`：主文件，包含数据采样、模型构建、训练、测试以及生成提交文件的完整流程。
2. `README.md`：本文件，包含本Baseline的简介以及使用方法。
3. `requirements.txt`：包含本Baseline的依赖库。
4. `test.csv`：测试数据集，与Kaggle上的一致
5. `.gitignore`：git忽略文件
6. `PINNs-组内专题.pptx`：PINNs的介绍PPT
7. `gt.csv`: 真实解数据集，用于计算误差
8. `evaluation.py`: 用于计算误差的函数

当运行本Baseline后会生成log文件夹，该文件夹中的文件介绍如下：
1. `model.pth`：训练好的模型，可以直接用于测试。
2. `baseline_submission.csv`：测试数据集的预测结果，可以直接提交到Kaggle。
3. `loss_PINN.txt`：训练过程中的损失函数日志，可以用于绘制损失函数曲线。
4. `loss_plot.png`：损失函数曲线，可以用于查看训练过程中的损失函数变化情况。
5. `Collocation Points.png`：训练过程中的采样点分布，可以用于查看采样点的分布情况。

这里需要额外介绍的是基础调参方法：
1. 模型：模型的层数、神经元数量、激活函数、模型结构等
2. 数据集：数据采样方法、数据集总数、mini-batch构造方法等
3. 训练：学习率、训练轮数、优化器、学习率衰减等
4. 损失函数：正则项、附加项等

希望大家在本次初赛中掌握PINN求解PDE的基础方法的同时还能够对调参有一定的了解。

最后：**Just Have Fun!**

## 利用PINNs求解PDE的基本流程
基本流程如下：

| 流程                  | 代码位置            |
|---------------------|-----------------|
| 构建初始条件、边界条件以及PDE数据集 | main.py 64-135  |
| 构建神经网络              | main.py 24-61   |
| 定义基于PDE的损失计算组件      | main.py 178-242 |
| 构建优化器等训练必要的组件       | main.py 165-171 |
| 构建并执行训练循环           | main.py 138-278 |
| 针对测试数据进行模型预测并保存结果   | main.py 281-301 |
| main流程              | main.py 304-341 |

## 运行方法以及提交指南

```commandline
# 切换到项目下
cd Baseline4SolvingSchrodingerEquation
# 训练并测试
python main.py
# 在服务器上如果需要指定显卡（示例为指定0卡，如果需要使用其他卡设置为其他数值），请使用
CUDA_VISIBLE_DEVICES=0 python main.py
# 提交文件将保存到log子文件下
# 将baseline_submission.csv文件提交到Kaggle就完成了一次提交
```

## 参考文献

-- Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational physics 378 (2019): 686-707.

-- Karniadakis, George Em, et al. "Physics-informed machine learning." Nature Reviews Physics 3.6 (2021): 422-440.

-- Cai, Shengze, et al. "Physics-informed neural networks (PINNs) for fluid mechanics: A review." Acta Mechanica Sinica 37.12 (2021): 1727-1738.

-- Yang, Zijiang, Zhongwei Qiu, and Dongmei Fu. "DMIS: dynamic mesh-based importance sampling for training physics-informed neural networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 4. 2023.

-- Lu, Lu, et al. "Physics-informed neural networks with hard constraints for inverse design." SIAM Journal on Scientific Computing 43.6 (2021): B1105-B1132.

-- Yu, Jeremy, et al. "Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems." Computer Methods in Applied Mechanics and Engineering 393 (2022): 114823.

-- Yang, Liu, Xuhui Meng, and George Em Karniadakis. "B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data." Journal of Computational Physics 425 (2021): 109913.

-- Jin, Xiaowei, et al. "NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations." Journal of Computational Physics 426 (2021): 109951.

-- Jagtap, Ameya D., Ehsan Kharazmi, and George Em Karniadakis. "Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems." Computer Methods in Applied Mechanics and Engineering 365 (2020): 113028.

-- Jagtap, Ameya D., Kenji Kawaguchi, and George Em Karniadakis. "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks." Journal of Computational Physics 404 (2020): 109136.
