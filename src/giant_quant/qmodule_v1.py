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

def pseudo_quantize_tensor(
    w, n_bit=4, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0, f'shape is {org_w_shape[-1]}'
        w = w.reshape(-1, q_group_size)
    # assert w.dim() == 2
    if w.dim() != 2:
        return w
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        # assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    w = w.float()
    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
    
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        name=None,
    ):
        # ctx: autograd.Function 提供的上下文对象，用于在前向传播过程中存储信息，以便在反向传播过程中使用。
        # quantize_elemwise_op 是求每个元素的数值，quantize_mx_op 应该是在求 scaling？

        use_quantize = False
        group_size = 32
        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        org_input_shape = input.shape
        org_weight_shape = weight.shape

        if bias is not None:
            ctx.has_bias = True
            # bf_bias = pseudo_quantize_tensor(bias, n_bit=4, zero_point=False, q_group_size=group_size)
        else:
            ctx.has_bias = False
            bf_bias = None

        if use_quantize:
            # element-wise quantize for input
            bf_in = pseudo_quantize_tensor(input, n_bit=4, zero_point=False, q_group_size=group_size)
            # element-wise quantize for weight and bias
            bf_weight = pseudo_quantize_tensor(weight, n_bit=4, zero_point=False, q_group_size=group_size)
            ctx.save_for_backward(bf_in, bf_weight)
            # compute output
            output = f_linear(bf_in, bf_weight)
            # print(bf_in.dtype, bf_weight.dtype, output.dtype)
            # output = pseudo_quantize_tensor(output, n_bit=4, zero_point=False, q_group_size=group_size)

        else:
            ctx.save_for_backward(input, weight)
            # init computation
            # with torch.cuda.amp.autocast(enabled=False):
            output = f_linear(input, weight)


        if bias is not None:
            if use_quantize:
                bf_bias = pseudo_quantize_tensor(bias, n_bit=4, zero_point=False, q_group_size=group_size)
                output = output + bf_bias
                output = pseudo_quantize_tensor(output, n_bit=4, zero_point=False, q_group_size=group_size)
            else:
                output = output + bias

        # print('forward')
        ctx.name = name

        # print(f"Forward pass - Input dtype: {input.dtype}, Weight dtype: {weight.dtype}, Output dtype: {output.dtype}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        input, weight = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        use_quantize = False
        group_size = 32
        org_input_shape = input.shape
        org_weight_shape = weight.shape


        # data type: grad_output is float32, input is float32, weight is float32
        grad_output = grad_output.to(dtype=torch.float32)
        input = input.to(dtype=torch.float32)
        weight = weight.to(dtype=torch.float32)

        grad_output = grad_output.reshape(-1, out_dim)
        input = input.reshape(-1, in_dim)

        if use_quantize:
            # grad_output = pseudo_quantize_tensor(grad_output.t(), n_bit=4, zero_point=False, q_group_size=group_size).t()
            # 对齐 qex_input 和 qex_grad_output 的类型，否则 matmul 无法计算；
            # input = input.half()

            # qex_input = pseudo_quantize_tensor(input.t(), n_bit=4, zero_point=False, q_group_size=group_size).t()
            # qex_grad_output = pseudo_quantize_tensor(grad_output.t(), n_bit=4, zero_point=False, q_group_size=group_size).t()

            qex_input = pseudo_quantize_tensor(input, n_bit=4, zero_point=False, q_group_size=group_size)
            qex_grad_output = pseudo_quantize_tensor(grad_output, n_bit=4, zero_point=False, q_group_size=group_size)

            qex_grad_output = qex_grad_output.reshape(-1, out_dim)
            qex_input = qex_input.reshape(-1, in_dim)
            # compute grad_weight [out_features, in_features]


            # Compute grad_weight -> fp32
            grad_weight = torch_matmul(qex_grad_output.transpose(0, 1), qex_input)
            grad_weight = pseudo_quantize_tensor(grad_weight, n_bit=4, zero_point=False, q_group_size=group_size)

                #####################################################
            # perform madtile operation for grad_input
            #####################################################
            # compute grad_input, quantize everything along output size
            qos_weight = pseudo_quantize_tensor(weight.t(), n_bit=4, zero_point=False, q_group_size=group_size).t()
            # qos_weight = qos_weight.half()

            # grad_output shape is (B, seq, out_dim)
            qos_grad_output = pseudo_quantize_tensor(grad_output, n_bit=4, zero_point=False, q_group_size=group_size)
            # qos_grad_output = qos_grad_output.half()

            # print(weight.shape, grad_output.shape)
            # print(qos_weight.dtype, qos_grad_output.dtype)
            # exit(0)

            # Compute grad_input
            grad_input = torch_matmul(qos_grad_output, qos_weight)
            grad_input = pseudo_quantize_tensor(grad_input, n_bit=4, zero_point=False, q_group_size=group_size)

        else:
            #####################################################
            # perform madtile operation for grad_weight, grad_bias
            #####################################################
            # if the input is 2D, quantize everything along examples (batches)
            # if the input is 3D, quantize everything along the first axis

            # reshape to 2D for transpose
            grad_weight = torch_matmul(grad_output.transpose(0, 1), input)

            # grad_weight = torch_matmul(input.transpose(0, 1), grad_output)

            # init computation
            grad_input = torch_matmul(grad_output, weight)


        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)
            if use_quantize:
                grad_bias = pseudo_quantize_tensor(grad_bias, n_bit=4, zero_point=False, q_group_size=group_size)

        grad_input = grad_input.reshape(org_input_shape)
        grad_weight = grad_weight.reshape(org_weight_shape)

        # print(f"Backward pass - Grad input dtype: {grad_input.dtype}, Grad weight dtype: {grad_weight.dtype}, Grad bias dtype: {grad_bias.dtype if grad_bias is not None else 'N/A'}")


        # grad_input = grad_input.to(dtype=torch.float16)
        # grad_weight = grad_weight.to(dtype=torch.float16)
        if grad_bias is not None:
            grad_bias = grad_bias.to(dtype=torch.float16)


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