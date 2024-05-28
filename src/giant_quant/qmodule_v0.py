import torch
import torch.nn as nn
import torch.autograd as autograd

class FakeQuantizeFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits):
        qmin = 0
        qmax = 2 ** num_bits - 1
        input = torch.clamp(torch.round(input / scale) + zero_point, qmin, qmax)
        ctx.save_for_backward(input, scale, zero_point)
        ctx.num_bits = num_bits
        return scale * (input - zero_point)

    @staticmethod
    def backward(ctx, grad_output):
        input, scale, zero_point = ctx.saved_tensors
        grad_input = grad_output.clone()  # This is a simple example, and you might need to modify it according to your needs
        return grad_input, None, None, None

def fake_quantize(x, scale, zero_point, num_bits):
    return FakeQuantizeFunction.apply(x, scale, zero_point, num_bits)

class FakeQuantizeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super(FakeQuantizeLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.num_bits = num_bits

    def forward(self, input):
        scale = 0.1  # 假量化的尺度因子
        zero_point = 0  # 假量化的零点

        # 前向传播时进行假量化
        weight_q = fake_quantize(self.linear.weight, scale, zero_point, self.num_bits)
        if self.linear.bias is not None:
            bias_q = fake_quantize(self.linear.bias, scale, zero_point, self.num_bits)
        else:
            bias_q = None

        return nn.functional.linear(input, weight_q, bias_q)