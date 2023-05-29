import time

import torch


def loss_fun(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def training():
    # 使用 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义参数模型
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    # 定义输入数据和真实标签
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y_ture = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

    # 定义学习率和迭代次数
    lr = 0.01
    num_epochs = 1000

    tic = time.time()
    # 开始训练
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = w * x + b

        # 计算损失
        loss = loss_fun(y_pred, y_ture)

        # 反向传播
        loss.backward()

        # 更新参数
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # 梯度清零
        w.grad.zero_()
        b.grad.zero_()

        if epoch % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss.item()}')

    toc = time.time()
    print(toc - tic)
    # 打印训练后的模型参数
    print('w: {:.4f}, b: {:.4f}'.format(w.item(), b.item()))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    v = ('supercali'  # '(' begins logical line,
         'fragilistic'  # indentation is ignored
         'expialidocious')  # until closing ')'
    print(f'Hi, {name}, {v}')
    # Press Ctrl+F8 to toggle the breakpoint.
    print(torch.__version__)

    print(torch.backends.cudnn.m.is_available())
    print(torch.cuda.is_available())

    # input_strings = ['1', '2', 'a', '11']
    # valid_int_strings = [int_s for s in input_strings
    #                      if (int_s := safe_int(s)) is not None]
    # print(valid_int_strings)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    if_cuda = torch.cuda.is_available()
    print("if_cuda=", if_cuda)

    gpu_count = torch.cuda.device_count()
    print("gpu_count=", gpu_count)

    training()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# 实现冒泡排序
def bubble_sort(arr):
    for i in range(len(arr) - 1):
        for j in range(len(arr) - 1 - i):  # 每一轮比较次数
            if arr[j] > arr[j + 1]:  # 如果前一个数大于后一个数
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # 交换位置
    return arr
