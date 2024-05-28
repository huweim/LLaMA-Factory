import torch
import torch.nn.functional as F
import torch.nn as nn

f_linear = F.linear
torch_matmul = torch.matmul

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, name=None):
        if bias is not None:
            ctx.has_bias = True
        else:
            ctx.has_bias = False
            bf_bias = None
        ctx.save_for_backward(input, weight, bias)

        output = f_linear(input, weight, bias)

        print(f"Forward pass - Input dtype: {input.dtype}, Weight dtype: {weight.dtype}, Output dtype: {output.dtype}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        org_input_shape = input.shape
        org_weight_shape = weight.shape

        grad_output = grad_output.float()
        input = input.float()

        grad_output = grad_output.reshape(-1, out_dim)
        input = input.reshape(-1, in_dim)

        print(f"Backward pass - Grad output dtype: {grad_output.dtype}, Input dtype: {input.dtype}, Weight dtype: {weight.dtype}")

        grad_weight = torch_matmul(grad_output.transpose(0, 1), input)
        grad_input = torch_matmul(grad_output, weight)

        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)

        grad_input = grad_input.reshape(org_input_shape)
        grad_weight = grad_weight.reshape(org_weight_shape)

        print(f"Backward pass - Grad input dtype: {grad_input.dtype}, Grad weight dtype: {grad_weight.dtype}, Grad bias dtype: {grad_bias.dtype if grad_bias is not None else 'N/A'}")

        return (grad_input, grad_weight, grad_bias, None)

# 自定义层
class QuantLinear(nn.Linear):
    def forward(self, inputs):
        return LinearFunction.apply(inputs, self.weight, self.bias)

# 输入数据
input = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)
weight = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)
bias = torch.randn(3, dtype=torch.float32, requires_grad=True)

# 自定义层
custom_layer = QuantLinear(3, 3)
custom_layer.weight = nn.Parameter(weight.clone().detach())
custom_layer.bias = nn.Parameter(bias.clone().detach())
custom_output = custom_layer(input)

# PyTorch内置层
torch_layer = nn.Linear(3, 3)
torch_layer.weight = nn.Parameter(weight.clone().detach())
torch_layer.bias = nn.Parameter(bias.clone().detach())
torch_output = torch_layer(input)

# 损失函数
loss_fn = nn.MSELoss()
target = torch.randn(3, 3)

# 计算损失
custom_loss = loss_fn(custom_output, target)
torch_loss = loss_fn(torch_output, target)

# 反向传播
custom_loss.backward()
torch_loss.backward()

# 打印结果
print("Custom layer results:")
print(f"Custom output: {custom_output}")
print(f"Custom loss: {custom_loss.item()}")
print(f"Custom input grad: {input.grad}")
print(f"Custom weight grad: {custom_layer.weight.grad}")
print(f"Custom bias grad: {custom_layer.bias.grad}")

print("\nTorch layer results:")
print(f"Torch output: {torch_output}")
print(f"Torch loss: {torch_loss.item()}")
print(f"Torch input grad: {input.grad}")
print(f"Torch weight grad: {torch_layer.weight.grad}")
print(f"Torch bias grad: {torch_layer.bias.grad}")

# 验证结果是否一致
print("\nVerification:")
print(f"Output close: {torch.allclose(custom_output, torch_output)}")
print(f"Loss close: {torch.allclose(torch.tensor(custom_loss.item()), torch.tensor(torch_loss.item()))}")
print(f"Input grad close: {torch.allclose(input.grad, input.grad)}")
print(f"Weight grad close: {torch.allclose(custom_layer.weight.grad, torch_layer.weight.grad)}")
print(f"Bias grad close: {torch.allclose(custom_layer.bias.grad, torch_layer.bias.grad)}")
