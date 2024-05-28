"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn.functional as F

# from .mx_ops import quantize_mx_op
# from .elemwise_ops import quantize_elemwise_op
# from .specs import apply_mx_specs, get_backwards_mx_specs
# from .specs import mx_assert_test

f_linear = F.linear
torch_matmul = torch.matmul

    
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        name=None,
    ):

        if bias is not None:
            ctx.has_bias = True
        else:
            ctx.has_bias = False
            bf_bias = None


        ctx.save_for_backward(input, weight)

        output = f_linear(input, weight)

        if bias is not None:
            output = output + bias

        ctx.name = name
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        input, weight = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]


        org_input_shape = input.shape
        org_weight_shape = weight.shape

        grad_output = grad_output.float()
        input = input.float()
        weight = weight.float()

        grad_output = grad_output.reshape(-1, out_dim)
        input = input.reshape(-1, in_dim)

        grad_weight = torch_matmul(grad_output.transpose(0, 1), input)

        grad_input = torch_matmul(grad_output, weight)

        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)

        grad_input = grad_input.reshape(org_input_shape)
        grad_weight = grad_weight.reshape(org_weight_shape)

        # print(grad_weight.dtype, grad_input.dtype, grad_bias.dtype if grad_bias is not None else None)


        return (grad_input, grad_weight, grad_bias, None, None, None, None)


def linear(
    input,
    weight,
    bias=None,
    name=None,
):
    return LinearFunction.apply(input, weight, bias, name)


class QuantLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        name=None,
    ):

        self.name = name
        super().__init__(in_features, out_features, bias)


    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):

        return linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            name=self.name,
        )