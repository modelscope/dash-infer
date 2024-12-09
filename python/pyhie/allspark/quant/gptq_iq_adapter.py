'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    gptq_iq_adapter.py
'''

import torch



def depack_zero(qzeros, w_bit=4):
    wf = torch.tensor(list(range(0, 32, w_bit)), dtype=torch.int32).to(qzeros.device).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // w_bit),
        wf.unsqueeze(0)
    ).to(torch.int16 if w_bit == 8 else torch.int8)
    torch.bitwise_and(zeros, (2 ** w_bit) - 1, out=zeros)
    zeros = zeros + 1
    zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
    return zeros


class GPTQ2IQWeightAdapter:
    def dequant_gptq_weight(self, model, w_bit, model_dtype):
        """
        dequant gptq weight to a float16/bfloat16 model
        Args:
            model: state_dict of pytorch module.
            w_bit: bits width

        Returns:

        """
        new_model = model.copy()
        for key, val in model.items():
            # print(f"process weight {key}  -- {val.dtype}")

            if key.endswith('qweight'):
                qweight = val
                zeros_key = key[:key.index('.qweight')] + ".qzeros"
                zeros = new_model[zeros_key]
                scales_key = key[:key.index('.qweight')] + ".scales"
                scales = new_model[scales_key]
                g_idx_key = key[:key.index('.qweight')] + ".g_idx"
                g_idx = new_model[g_idx_key]
                group_size = g_idx.size()[0] // scales.size()[0]
                weight = self.dequantize_tensor(self.depack_weight(qweight, w_bit), scales, depack_zero(zeros, w_bit), group_size)
                weight = weight.t().contiguous()

                weight = self.valid_weight_data_type(key, model_dtype, weight)
                new_model[key.replace('qweight', 'weight')] = weight

                # print(f"remove {key} {zeros_key} {scales_key} {g_idx_key} add {key.replace('qweight', 'weight')}")
                del new_model[key]
                del new_model[zeros_key]
                del new_model[scales_key]
                del new_model[g_idx_key]
            else:
                new_model[key] = self.valid_weight_data_type(key, model_dtype, val)


        return new_model

    def valid_weight_data_type(self, key, model_dtype, weight):
        if (model_dtype == 'bfloat16' and weight.dtype == torch.float16):
            print(
                f"Warn: weight: {key} model data type:{model_dtype} and weight data:{weight.dtype} type different, convert weight to {model_dtype}, model accuracy may drop.")
            weight = weight.bfloat16()
        elif (model_dtype == 'float16' and weight.dtype == torch.bfloat16):
            print(
                f"Warn: weight: {key} model data type:{model_dtype} and weight data:{weight.dtype} type different, convert weight to {model_dtype}, model accuracy may drop.")
            weight = weight.half()
        return weight

    def depack_weight(self, qweight, bits=4):
        wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).to(qweight.device).unsqueeze(0)
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
        return weight

    def dequantize_tensor(self, weight, scales, zeros, group_size):
        scales = scales.reshape(-1, 1, scales.shape[-1])
        zeros = zeros.reshape(-1, 1, zeros.shape[-1])
        weight = weight.reshape(-1, group_size, weight.shape[2])
        weight = scales * (weight - zeros)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        return weight