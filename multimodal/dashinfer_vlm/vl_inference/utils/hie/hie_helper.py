'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    hie_helper.py
'''
import os
import onnxruntime
import onnx
import logging
import numpy as np


def get_data_type(data_type):
    return np.dtype(data_type)


def numpy_to_hie_txt(np_data_dict, file):
    with open(file, "w") as f:
        for name, data in np_data_dict.items():
            f.write(name + " ")
            f.write(str(data.dtype) + " ")
            f.write("x".join(map(str, data.shape)) + " ")
            if data.dtype == np.dtype(bool):
                f.write(",".join(map(str, 1 * data.flatten())))
            else:
                f.write(",".join(map(str, data.flatten())))
            f.write("\n")


def numpy_to_hie_npz(np_data_dict, file):
    np.savez(file, **np_data_dict)


def numpy_to_hie_npy(np_data_dict, dirname):
    for name, data in np_data_dict.items():
        filename = os.path.join(dirname, name.replace("/", "_") + ".npy")
        np.save(filename, data)


def load_hie_data_txt(file):
    data = {}
    with open(file, "r") as f:
        array = f.readlines()
        for line in array:
            line = line.strip().split(" ")
            name = line[0]
            shape = list(map(int, line[2].strip().split("x")))
            # dtype = get_data_type(line[1])
            dtype = np.dtype(line[1])
            str_list = line[3].strip().split(",")
            if dtype == "bool":
                value = np.array(str_list).astype(int).reshape(shape) > 0
            else:
                value = np.array(str_list).astype(dtype).reshape(shape)
            data[name] = value
    return data


def load_hie_data_npz(file):
    npz = np.load(file)
    return dict(zip(npz.keys(), npz.values()))


def load_hie_data_npy(file):
    data_npy = np.load(file)
    return {"tensor": data_npy}


# OnnxRuntime Runner
# run onnx model by onnxruntime


def run_ort(
    onnx_model_fn,
    onnx_model,
    input_values,
    output_names=None,
    providers=["CPUExecutionProvider"],
):
    """

    :param onnx_model:
    :param input_values:
    :param output_names:
    :return:
    """
    old_output = [n.name for n in onnx_model.graph.output]
    if output_names is None:
        output_names = old_output
    logging.info("  inputs: {}".format(input_values.keys()))
    logging.info("  outputs: {}".format(output_names))
    for n in output_names:
        if n not in old_output:
            it_have_vi = False
            for vi in onnx_model.graph.value_info:
                if vi.name == n:
                    onnx_model.graph.output.append(vi)
                    it_have_vi = True
                    break
            if it_have_vi is False:
                onnx_model.graph.output.append(
                    onnx.helper.make_empty_tensor_value_info(n)
                )

    session = onnxruntime.InferenceSession(onnx_model_fn, providers=providers)
    onnx_outputs = session.run(output_names, input_values)
    session.end_profiling()
    return dict(zip(output_names, onnx_outputs))


def get_input_values(inputs=None, inputs_preset=None, onnx_model=None):
    input_values = {}
    if inputs is not None:
        names, shapes, dtypes = [], [], []
        for i in inputs:
            names.append(i[0])
            shapes.append(i[1])
            dtypes.append(i[2])
        for i, name in enumerate(names):
            if shapes[i][0] == -1:
                shapes[i][0] = 1
            if "float" in dtypes[i]:
                input_values[name] = np.random.rand(*shapes[i]).astype(dtypes[i])
            elif "int" in dtypes[i]:
                input_values[name] = np.random.randint(
                    low=0, high=100, size=shapes[i]
                ).astype(dtypes[i])
            else:
                raise NotImplementedError
    elif inputs_preset is not None:
        input_values = inputs_preset
    elif onnx_model is not None:
        inputs = onnx_model.graph.input
        initializer_names = set([i.name for i in onnx_model.graph.initializer])
        for i in inputs:
            if i.name in initializer_names:
                continue
            shape = [
                d.dim_value if d.dim_value > 0 else 1
                for d in i.type.tensor_type.shape.dim
            ]
            print(f"use random input: {i.name}, {shape}")
            input_values[i.name] = np.random.rand(*shape).astype(
                onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type]
            )
    else:
        raise ValueError("Input values can not be prepared.")
    return input_values


def profile_ort(
    onnx_model_fn,
    onnx_model,
    input_values,
    run_times,
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
):
    """

    :param onnx_model:
    :param input_values:
    :return:
    """
    inputs = [n.name for n in onnx_model.graph.input]
    outputs = [n.name for n in onnx_model.graph.output]
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    logging.info("  inputs: {}".format(inputs))
    logging.info("  outputs: {}".format(outputs))

    session = onnxruntime.InferenceSession(
        onnx_model_fn, providers=providers, sess_options=options
    )

    for _ in range(run_times):
        session.run(outputs, input_values)

    profile_file = session.end_profiling()

    import json
    import os

    dur = 0
    with open(profile_file, "r") as f:
        sess_time = json.load(f)
    os.remove(profile_file)
    if not sess_time:
        logging.error("Invalid json")
        return 0
    for record in sess_time:
        if record["name"] == "model_run":
            dur += record["dur"]
    return dur / run_times


def checker(data_0, data_1):
    """

    :param data_0: numpy.array
    :param data_1: numpy.array
    :return diff_max, diff_mean, diff_var, rmse:
    """
    print("  shape: {}".format(data_0.shape))
    if data_0.dtype == np.dtype("bool"):
        diff = (data_0 == data_1).all()
        print("  allsame: {}".format(diff))
        return
    else:
        data_0_nan = np.argwhere(np.isnan(data_0))
        data_1_nan = np.argwhere(np.isnan(data_1))
        if data_0_nan.size != 0 or data_1_nan.size != 0:
            print("  a's nan: {}".format(data_0_nan))
            print("  b's nan: {}".format(data_1_nan))
            return
        data_0 = data_0.astype(np.float64)
        data_1 = data_1.astype(np.float64)
        diff = data_0 - data_1
        diff_0 = diff.reshape(-1)
        data_0_ = data_0.reshape(-1)
        data_1_ = data_1.reshape(-1)
        diff_abs = abs(diff_0)
        diff_abs_max_index = np.argmax(diff_abs)
        diff_abs_max = diff_abs[diff_abs_max_index]
        with np.errstate(invalid="ignore"):
            diff_rel = np.where(data_1_ != 0, diff_abs / abs(data_1_), 10000)
        diff_rel_max_index = np.argmax(diff_rel)
        diff_rel_max = diff_rel[diff_rel_max_index]
        diff_mean = np.mean(diff_0)
        diff_var = np.var(diff_0)
        rmse = np.sqrt((diff**2).mean())
        cos_sim = np.dot(data_0_, data_1_) / (
            np.linalg.norm(data_0_) * np.linalg.norm(data_1_)
        )
        print(
            "  diff_abs_max: {}, where a = {}, b = {}".format(
                diff_abs_max, data_0_[diff_abs_max_index], data_1_[diff_abs_max_index]
            )
        )
        print(
            "  diff_rel_max: {}, where a = {}, b = {}".format(
                diff_rel_max, data_0_[diff_rel_max_index], data_1_[diff_rel_max_index]
            )
        )
        print("  diff_mean: {}".format(diff_mean))
        print("  diff_var: {}".format(diff_var))
        print("  rmse: {}".format(rmse))
        print("  cos_sim: {}".format(cos_sim))
        return
