'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    quantization_utils.py
'''
from .model_base import *
from ..quantization import QuantizeConfig
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
    elif dtype == "FP8_E4M3":
        return torch.float8_e4m3fn
    else:
        raise Exception("DataType Error")

def get_groupsize_by_quant_config(quant_config, torch_name = None) -> int:
    if not quant_config.extra_option["SubChannel"]:
        raise ValueError("GroupSize is only set in SubChannel mode")

    group_size = -1
    if quant_config.extra_option.get("GroupSettings", None) is None:
        group_size = int(quant_config.extra_option["GroupSize"])
    else:
        if torch_name == None:
            raise Exception("GroupSettings match require torch name.")
        match_flag = False
        for key, val in quant_config.extra_option["GroupSettings"].items():
            if re.match(r'^.*\b%s.*$' % key, torch_name, re.I):
                group_size = int(val)
                match_flag = True
                break
        if not match_flag:
            raise ValueError(f"Original weight name: {torch_name} not match any regex in GroupSettings!")  
    return group_size

def quantize_gemm(op, quant_config, orig_weight_name = None):
    if quant_config.quantize_mode in [
            QuantizeConfig.QuantMode.A16W8, QuantizeConfig.QuantMode.A16W4, QuantizeConfig.QuantMode.A8W8, QuantizeConfig.QuantMode.FP8A8W8
    ]:
        quant_op_type = "GemmA16W8"
        if quant_config.quantize_mode == QuantizeConfig.QuantMode.A16W8:
            quant_op_type = "GemmA16W8"
        elif quant_config.quantize_mode == QuantizeConfig.QuantMode.A16W4:
            quant_op_type = "GemmA16W4"
        elif quant_config.quantize_mode == QuantizeConfig.QuantMode.A8W8:
            quant_op_type = "GemmA8W8"
        else:
            quant_op_type = "GemmFP8A8W8"
        if op.op_type.upper() == "GEMM":
            op.op_type = quant_op_type
        else:
            op.attr["InnerGemmType"] = quant_op_type.encode()
        op.weights.insert(1, make_tensor(op.op_name + ".weight.scale"))
        op.weights.insert(2, make_tensor(op.op_name + ".weight.zero_point"))
        # set attr GroupSize
        if quant_config.extra_option["SubChannel"]:
            group_size = get_groupsize_by_quant_config(quant_config, orig_weight_name)
            op.attr["GroupSize"] = np.array(group_size).astype(
                "int32").tobytes()
            # print(orig_weight_name, op.weights[0].name, group_size)
    return op

def quantize_moe(op, quant_config, orig_weight_name = None):
    if quant_config.quantize_mode in [
            QuantizeConfig.QuantMode.A16W8, QuantizeConfig.QuantMode.A16W4, QuantizeConfig.QuantMode.A8W8,
    ]:
        # only support moe a8w8
        quant_op_type = "MOEA8W8"
        op.op_type = quant_op_type
        op.weights.insert(1, make_tensor(op.op_name + ".gate_up_proj.weight.scale"))
        op.weights.insert(2, make_tensor(op.op_name + ".gate_up_proj.weight.zero_point"))
        op.weights.insert(4, make_tensor(op.op_name + ".down_proj.weight.scale"))
        op.weights.insert(5, make_tensor(op.op_name + ".down_proj.weight.zero_point"))
    return op
def quantize_op(op, quant_config, quantize_map, weight_name_map = None):
    if quant_config == None:
        return op
    try:
        if op.op_type.upper().startswith("GEMM") and op.weights[
                0].name in quantize_map:  # compatible with GemmCapsule
            if weight_name_map:
                orig_weight_name = weight_name_map[op.weights[0].name]
                return quantize_gemm(op, quant_config, orig_weight_name)
            else:
                return quantize_gemm(op, quant_config)
        if op.op_type.upper().startswith("MOE") and op.weights[
                0].name in quantize_map:  # compatible with GemmCapsule
            if weight_name_map:
                orig_weight_name = weight_name_map[op.weights[0].name]
                return quantize_moe(op, quant_config, orig_weight_name)
            else:
                return quantize_moe(op, quant_config)
    except Exception as e:
        raise e


###################################

def quantize_gemm_weight_fp8_torch(fdata,
                                     quant_config,
                                     torch_tensor_info=None):
    if isinstance(fdata, torch.Tensor) == False:
        raise Exception("Quantize GemmFP8 weight type error.")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    fdata = fdata.to(device_type)
    qtype = convert_dtype_to_torch(quant_config.weight_type)

    # qmax = torch.tensor(torch.finfo(qtype).max,
    #                     dtype=torch.float32,
    #                     device=device_type)
    # qmin = torch.tensor(torch.finfo(qtype).min,
    #                     dtype=torch.float32,
    #                     device=device_type)

    # K = fdata.shape[0]
    # N = fdata.shape[1]

    # # Find Max-Min
    # fmax = torch.amax(torch.abs(fdata), keepdim=True).squeeze(dim=-1).to(torch.float32)
    # # Compute params
    # scale = fmax / qmax
    # zero = torch.zeros(1, dtype=torch.float32, device=device_type)
    # # Quantize
    # res_data = fdata / scale + zero
    # qdata = torch.clamp(res_data.float(), qmin, qmax)

    # qdata = qdata.to(qtype).cpu()
    # scale = scale.cpu()
    # zero = zero.cpu()
    # print(f"qdata : {qdata}, qdata_shape : {qdata.shape}, scale : {scale}, scale_shape : {scale.shape}, zero : {zero}, zero_shape : {zero.shape}")

    # scale and zero are used as placeholders first, and the actual calculation is done in the operator init phase.
    # This can achieve the card-splitting per-tensor fp8 quantization in the case of multiple cards
    scale = torch.zeros(1, dtype=torch.float32, device=device_type)
    zero = torch.zeros(1, dtype=torch.float32, device=device_type)
    return fdata, scale, zero

def quantize_gemm_weight_a16w8_torch(fdata,
                                     quant_config,
                                     torch_tensor_info=None):
    if isinstance(fdata, torch.Tensor) == False:
        raise Exception("Quantize GemmA16W8 weight type error.")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    fdata = fdata.to(device_type)
    ftype = fdata.dtype
    qtype = convert_dtype_to_torch(quant_config.weight_type)

    if quant_config.extra_option.get("AdaptedQuantMethod") == "GPTQ":
        if torch_tensor_info == None:
            raise Exception("gptq require torch info.")
        return repack_gptq_to_a16wX(torch_tensor_info, 8)
    elif quant_config.extra_option.get("AdaptedQuantMethod") == "GPTQ_NO_PACK":
        return repack_gptq_no_pack_to_a16wX(torch_tensor_info, 8)

    qmax = torch.tensor(torch.iinfo(qtype).max,
                        dtype=torch.float32,
                        device=device_type)
    qmin = torch.tensor(torch.iinfo(qtype).min,
                        dtype=torch.float32,
                        device=device_type)

    K = fdata.shape[0]
    N = fdata.shape[1]

    if torch_tensor_info != None:
        w_torch_name = torch_tensor_info[0]
    else:
        w_torch_name = None
    GroupSize = get_groupsize_by_quant_config(quant_config, w_torch_name) if quant_config.extra_option["SubChannel"] == True else K
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


def get_4bit_quant_maxmin(dtype):
    dtype = dtype.upper()
    if dtype == "INT4":
        return int(7), int(-8)
    elif dtype == "UINT4":
        return int(15), int(0)
    else:
        raise Exception("4Bit data-type error")


def get_4bit_pack_dtype_torch(dtype):
    dtype = dtype.upper()
    if dtype == "INT4":
        return torch.int8
    elif dtype == "UINT4":
        return torch.uint8
    else:
        raise Exception("4Bit data-type error")


def quantize_gemm_weight_a16w4_torch(fdata,
                                     quant_config,
                                     torch_tensor_info=None):
    if isinstance(fdata, torch.Tensor) == False:
        raise Exception("Quantize GemmA16W8 weight type error.")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    #device_type = "cpu"
    # force use cpu to avoid cuda memory already allocated., maybe slow...
    fdata = fdata.to(device_type)
    ftype = fdata.dtype
    if quant_config.extra_option.get("AdaptedQuantMethod") == "GPTQ":
        if torch_tensor_info == None:
            raise Exception("gptq require torch tensor info")
        return repack_gptq_to_a16wX(torch_tensor_info, 4)

    qmax, qmin = get_4bit_quant_maxmin(quant_config.weight_type)
    qmax = torch.tensor(qmax, dtype=torch.float32, device=device_type)
    qmin = torch.tensor(qmin, dtype=torch.float32, device=device_type)

    K = fdata.shape[0]
    N = fdata.shape[1]

    # If SubChannel Pad K to GroupSize
    if torch_tensor_info != None:
        w_torch_name = torch_tensor_info[0]
    else:
        w_torch_name = None
    GroupSize = get_groupsize_by_quant_config(quant_config, w_torch_name) if quant_config.extra_option["SubChannel"] == True else K
    KStride = int((K + GroupSize - 1) / GroupSize) * int(GroupSize)
    KPad = KStride - K
    pad_k_val = fdata[-1, :].view(1, N).repeat(KPad, 1)
    fdata_pad = torch.cat((fdata, pad_k_val), -2)
    # Pad N to 2
    NMod = 2
    NStride = int((N + NMod - 1) / NMod) * int(NMod)
    NPad = NStride - N
    fdata_pad = torch.nn.ConstantPad2d((0, NPad, 0, 0), 0)(fdata_pad)

    data = torch.transpose(fdata_pad, 1, 0).reshape(NStride, -1, GroupSize)
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

    qdata = torch.transpose(qdata.view(NStride, -1), 1, 0).contiguous()
    scale = torch.transpose(scale.view(NStride, -1), 1, 0).contiguous()
    zero = torch.transpose(zero.view(NStride, -1), 1, 0).contiguous()

    # Pack
    qdata = qdata.to(get_4bit_pack_dtype_torch(quant_config.weight_type))
    qdata_pack = (qdata[:, 1::2] << 4) | (qdata[:, 0::2] & 0xf)

    qdata_pack = qdata_pack[:K, :].cpu()
    scale = scale[:, 0:N].to(ftype).cpu()
    zero = zero[:, 0:N].to(ftype).cpu()

    return qdata_pack, scale, zero


def quantize_gemm_weight_a16wX_torch(fdata,
                                     quant_config,
                                     torch_tensor_info=None):
    # both a16w8 and a8w8 share the same quantize function
    ret = []
    if quant_config.quantize_mode in [
        QuantizeConfig.QuantMode.A16W8,
        QuantizeConfig.QuantMode.A8W8,
    ]:
        ret = quantize_gemm_weight_a16w8_torch(fdata, quant_config,
                                                torch_tensor_info)
    elif quant_config.quantize_mode in [QuantizeConfig.QuantMode.A16W4]:
        ret =  quantize_gemm_weight_a16w4_torch(fdata, quant_config,
                                                torch_tensor_info)
    elif quant_config.quantize_mode in [QuantizeConfig.QuantMode.FP8A8W8]:
        ret =  quantize_gemm_weight_fp8_torch(fdata, quant_config,
                                                torch_tensor_info)
    else:
        raise Exception("A16WX Quantize Error.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ret


def depack_gptq_weight(qweight, bits=4):
    wf = torch.tensor(list(range(0, 32, bits)),
                      dtype=torch.int32).to(qweight.device).unsqueeze(0)
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
    ## WARN: this script will convert to torch int16 if in 8bit quant, and int8 in 4bit  model
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)
    return weight


def depack_gptq_zero(qzeros, w_bit=4):
    wf = torch.tensor(list(range(0, 32, w_bit)),
                      dtype=torch.int32).to(qzeros.device).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // w_bit),
        wf.unsqueeze(0)).to(torch.int16 if w_bit == 8 else torch.int8)
    torch.bitwise_and(zeros, (2**w_bit) - 1, out=zeros)
    zeros = zeros + 1
    zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
    return zeros

def repack_gptq_no_pack_to_a16wX(torch_tensor_info, quant_bits, transpose=True):
    w_name = torch_tensor_info[0]
    torch_tensor = torch_tensor_info[1]
    assert w_name.rfind('.weight') != -1
    stem_name = re.sub(r'\.weight$', '', w_name)
    # qweight_name = re.sub(r'\.weight$', '.qweight', w_name)
    qzeros_name = f"{stem_name}.qzeros"
    scales_name = f"{stem_name}.scales"
    scales = torch_tensor[scales_name]
    if transpose:
        scales = scales.t()
    ftype = scales.dtype
    if quant_bits == 8:
        weight_type = torch.int8
    else:
        weight_type = None 
        raise ValueError(f"not supported quant_bits: {quant_bits} for per channel quantization")
    w = torch_tensor[w_name] 

    assert len(w.shape) == 2
    assert w.dtype in [torch.float16, torch.int16]
    qdata = w.to(weight_type)
    if transpose:
        qdata = qdata.t().contiguous()

    w = torch_tensor.get(qzeros_name)
    qzeros = None
    if w is not None:
        assert len(w.shape) == 2
        qzeros = w.to(ftype)
        if transpose:
            qzeros = qzeros.t()
    else:
        qzeros = torch.zeros_like(scales)
        print(f"warning: qzeros not found, use zeros instead, qzero shape: {qzeros.shape}, qzeros dtype: {qzeros.dtype} scales shape:{scales.shape}, scale dtype:{scales.dtype}")

    return qdata, scales, qzeros

def repack_gptq_to_a16wX(torch_tensor_info, quant_bits):
    w_name = torch_tensor_info[0]
    torch_tensor = torch_tensor_info[1]
    assert w_name.rfind('.weight') != -1
    stem_name = re.sub(r'\.weight$', '', w_name)
    # qweight_name = re.sub(r'\.weight$', '.qweight', w_name)
    qzeros_name = f"{stem_name}.qzeros"
    scales_name = f"{stem_name}.scales"
    # g_idx_name = f"{stem_name}.g_idx"
    scales = torch_tensor[scales_name]
    ftype = scales.dtype

    # quant qweight
    w = torch_tensor[w_name]
    assert len(w.shape) == 2
    qdata = None
    if w.dtype == torch.int8:
        qdata = w
    else:
        qdata = depack_gptq_weight(w, quant_bits)
    qdata = qdata.reshape(-1, qdata.shape[-1])

    if quant_bits == 4:
        assert (qdata
                >= 0).all()  # depack后的qdata以int8存储, 需要模型提供方保证实际是以uint4量化的
        qdata = qdata.to(torch.uint8)  # positive int8 -> uint8
        assert qdata.shape[1] % 2 == 0
        qdata_pack = (qdata[:, 1::2] << 4) | (qdata[:, 0::2] & 0xf)
        assert w.shape[0] * 8 == qdata_pack.shape[0]
        assert qdata_pack.shape[1] * 2 == w.shape[1]
    elif quant_bits == 8:
        qdata_pack = qdata
        ## XXX: force convert qdata to int8, kernel only support int8.
        qdata_pack = qdata_pack.to(torch.int8)
    else:
        raise ValueError("not supported quant_bits: {}".format(quant_bits))
    # quant qzeros
    w = torch_tensor.get(qzeros_name)
    qzeros = None
    if w is not None:
        assert len(w.shape) == 2
        qzeros = depack_gptq_zero(w, quant_bits).to(ftype)
    else:
        qzeros = torch.zeros_like(scales)
        print(f"warning: qzeros not found, use zeros instead, qzero shape: {qzeros.shape}, qzeros dtype: {qzeros.dtype} scales shape:{scales.shape}, scale dtype:{scales.dtype}")

    return qdata_pack, scales, qzeros
