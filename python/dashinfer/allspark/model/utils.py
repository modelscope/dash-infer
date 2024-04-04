#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    utils.py
#
import re
import numpy as np


class WeightNameAdapter:

    def __init__(self, std_names, real_names, pattern_rules=None):
        self.real_names = real_names
        if pattern_rules is None:
            pattern_rules = {
                0: r"\b%s\b",
                4: r"\blayers\.\d+\..*\b%s\b",
            }

        # gen real_name regex rules for getting prefix and body
        name_patterns = []
        start_pos_arr = list(pattern_rules.keys())
        for i in range(len(start_pos_arr)):
            start_pos = start_pos_arr[i]
            end_pos = len(std_names) if i + 1 == len(
                start_pos_arr) else start_pos_arr[i + 1]
            for j in range(start_pos, end_pos):
                name_patterns.append(pattern_rules[start_pos_arr[i]] %
                                     std_names[j])

        # split real_name into prefix & body according std_name
        segments_dict = {}
        for name in real_names:
            for i in range(len(name_patterns)):
                m = re.match(r'^(.*)\b(%s.*)$' % name_patterns[i], name, re.I)
                if m:
                    if segments_dict.get(std_names[i]):
                        break
                    body = m.group(2)
                    if name_patterns[i].startswith(r'\blayers\.\d+\.'):
                        body = re.sub(r'^layers\.\d+\.', 'layers.{}.', body)
                    elif name_patterns[i].startswith(r'\blayer\.\d+\.'):
                        body = re.sub(r'^layer\.\d+\.', 'layer.{}.', body)
                    elif name_patterns[i].startswith(r'\bh\.\d+\.'):
                        body = re.sub(r'^h\.\d+\.', 'h.{}.', body)
                    elif name_patterns[i].startswith(
                            r'\btransformer\.h\.\d+\.'):
                        body = re.sub(r'^transformer\.h\.\d+\.',
                                      'transformer.h.{}.', body)
                    segments_dict[std_names[i]] = [m.group(1), body]
                    break
        self.weight_name_segments = segments_dict

    def fullname(self, std_name):
        return self.weight_name_segments[std_name][
            0] + self.weight_name_segments[std_name][1]

    def origname(self, full_name):
        for real_name in self.real_names:
            if full_name.rfind(real_name) != -1:
                return real_name
        raise ValueError(
            f"Original name of {full_name} not found in torch model keys!")


def quantiza_data_by_numpy(data, dtype="int8", axis=None):
    """
    Quantize data by numpy
    Args:
        data:
        dtype:
        axis:

    Returns:

    """
    qmax = float(np.iinfo(dtype).max)
    qmin = float(np.iinfo(dtype).min)

    fmax = np.amax(data, axis=axis, keepdims=True)
    fmin = np.amin(data, axis=axis, keepdims=True)

    scale = (fmax - fmin) / (qmax - qmin)
    init_zero = qmin - fmin / scale
    init_zero = init_zero.astype("float32")
    np.where(init_zero >= qmin, init_zero, qmin)
    np.where(init_zero <= qmax, init_zero, qmax)
    zero_point = np.round(init_zero).astype(dtype)

    qdata = data / scale + init_zero
    qdata = np.round(np.clip(qdata, qmin, qmax)).astype(dtype)
    scale = scale.astype("float32")
    redsum = np.sum(qdata, axis=axis, dtype="int32", keepdims=True)

    return qdata, scale, init_zero, redsum
