import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantLinear(nn.Linear):
    def forward(self, inputs):
        return LinearFunction.apply(inputs, self.weight, self.bias)

# PyTorch内置的Linear层
class TorchLinear(nn.Linear):
    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)

# 自定义的前向和反向传播函数
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, name=None):
        ctx.save_for_backward(input, weight, bias)
        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output.mm(weight.t())
        grad_weight = input.t().mm(grad_output)
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None

# 测试自定义层和PyTorch内置层的输出和梯度
input = torch.randn(3, 3, requires_grad=True)
weight = torch.randn(3, 3, requires_grad=True)
bias = torch.randn(3, requires_grad=True)

custom_layer = QuantLinear(3, 3)
torch_layer = TorchLinear(3, 3)

# 将权重和偏置赋值给自定义层和内置层
custom_layer.weight = nn.Parameter(weight.clone().detach())
custom_layer.bias = nn.Parameter(bias.clone().detach())
torch_layer.weight = nn.Parameter(weight.clone().detach())
torch_layer.bias = nn.Parameter(bias.clone().detach())

# 计算输出
custom_output = custom_layer(input)
torch_output = torch_layer(input)

# 计算损失
loss_fn = nn.MSELoss()
target = torch.randn(3, 3)
custom_loss = loss_fn(custom_output, target)
torch_loss = loss_fn(torch_output, target)

# 反向传播
custom_loss.backward()
torch_loss.backward()

# 打印结果
print(f"Custom output: {custom_output}")
print(f"Torch output: {torch_output}")
print(f"Custom loss: {custom_loss.item()}")
print(f"Torch loss: {torch_loss.item()}")
print(f"Custom input grad: {input.grad}")
print(f"Torch input grad: {input.grad}")
print(f"Custom weight grad: {custom_layer.weight.grad}")
print(f"Torch weight grad: {torch_layer.weight.grad}")
print(f"Custom bias grad: {custom_layer.bias.grad}")
print(f"Torch bias grad: {torch_layer.bias.grad}")