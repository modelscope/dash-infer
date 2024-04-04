#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    quantization_utils.py
#
from .model_base import *
from dashinfer.allspark.quantization import QuantizeConfig
import numpy as np
import torch


def convert_dtype_to_torch(dtype):
    dtype = dtype.upper()
    if dtype == "FLOAT":
        return torch.float
    elif dtype == "DOUBLE":
        return torch.double
    elif dtype == "FLOAT16":
        return torch.float16
    elif dtype == "BFLOAT16":
        return torch.bfloat16
    elif dtype == "UINT8":
        return torch.uint8
    elif dtype == "INT8":
        return torch.int8
    elif dtype == "INT16":
        return torch.int16
    elif dtype == "INT32":
        return torch.int32
    elif dtype == "INT64":
        return torch.int64
    else:
        raise Exception("DataType Error")


def quantize_gemm(op, quant_config):
    if quant_config.quantize_mode in [QuantizeConfig.QuantMode.A16W8]:
        quant_op_type = "GemmA16W8"
        if op.op_type.upper() == "GEMM":
            op.op_type = quant_op_type
        else:
            op.attr["InnerGemmType"] = quant_op_type.encode()
        op.weights.insert(1, make_tensor(op.op_name + ".weight.scale"))
        op.weights.insert(2, make_tensor(op.op_name + ".weight.zero_point"))
        if quant_config.extra_option["SubChannel"]:
            group_size = int(quant_config.extra_option["GroupSize"])
            op.attr["GroupSize"] = np.array(group_size).astype(
                "int32").tobytes()

    return op


def quantize_op(op, quant_config, quantize_map):
    if quant_config == None:
        return op
    try:
        if op.op_type.upper().startswith("GEMM") and op.weights[
                0].name in quantize_map:  # compatible with GemmCapsule
            return quantize_gemm(op, quant_config)
    except Exception as e:
        raise e


###################################


def quantize_gemm_weight_a16w8_torch(fdata, quant_config):
    if isinstance(fdata, torch.Tensor) == False:
        raise Exception("Quantize GemmA16W8 weight type error.")
    device_type = "cpu"
    fdata = fdata.to(device_type)
    ftype = fdata.dtype
    qtype = convert_dtype_to_torch(quant_config.weight_type)

    qmax = torch.tensor(torch.iinfo(qtype).max,
                        dtype=torch.float32,
                        device=device_type)
    qmin = torch.tensor(torch.iinfo(qtype).min,
                        dtype=torch.float32,
                        device=device_type)

    K = fdata.shape[0]
    N = fdata.shape[1]
    GroupSize = quant_config.extra_option[
        "GroupSize"] if quant_config.extra_option["SubChannel"] == True else K
    KStride = int((K + GroupSize - 1) / GroupSize) * int(GroupSize)
    KPad = KStride - K

    pad_val = fdata[-1, :].view(1, N)
    pad_val = pad_val.repeat(KPad, 1)
    fdata_pad = torch.cat((fdata, pad_val), -2)
    data = torch.transpose(fdata_pad, 1, 0).reshape(N, -1, GroupSize)

    # Find Max-Min
    fmax = torch.amax(data, dim=-1, keepdim=True).to(torch.float32)
    fmin = torch.amin(data, dim=-1, keepdim=True).to(torch.float32)
    # Compute params
    scale = (fmax - fmin) / (qmax - qmin)
    scale = torch.where(
        scale == 0, torch.tensor(1, dtype=torch.float32, device=device_type),
        scale)
    zero = qmin - fmin / scale
    # Quantize
    res_data = data / scale + zero
    qdata = torch.round(torch.clamp(res_data.float(), qmin, qmax))

    qdata = torch.transpose(qdata.view(N, -1), 1,
                            0).contiguous().to(qtype).cpu()
    scale = torch.transpose(scale.view(N, -1), 1,
                            0).contiguous().to(ftype).cpu()
    zero = torch.transpose(zero.view(N, -1), 1, 0).contiguous().to(ftype).cpu()
    qdata = qdata[0:K, :]
    return qdata, scale, zero


def quantize_gemm_weigt_a16wX_torch(fdata, quant_config):
    if quant_config.quantize_mode == QuantizeConfig.QuantMode.A16W8:
        return quantize_gemm_weight_a16w8_torch(fdata, quant_config)
    else:
        raise Exception("A16WX Quantize Error.")
