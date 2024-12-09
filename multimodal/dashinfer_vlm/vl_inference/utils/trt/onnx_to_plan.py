# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import Any, Dict, List, Optional
import contextlib
from dataclasses import dataclass

import tensorrt as trt
import torch

from .. import Qwen2VisionTransformer


class ONNX_TRT:

    def __init__(self, model_path=None):
        from transformers.models.qwen2_vl.configuration_qwen2_vl import (
            Qwen2VLVisionConfig,
        )

        self.model_path = model_path
        self.config = Qwen2VLVisionConfig.from_pretrained(
            model_path, trust_remote_code=True, revision=None, code_revision=None
        )
        self.input_embed_dim = (
            self.config.in_channels
            * self.config.temporal_patch_size
            * self.config.patch_size
            * self.config.patch_size
        )

    def export_onnx(self, onnx_file_path):
        print("Start converting ONNX model!")

        # class SumModule(torch.nn.Module):
        #     def forward(self, x, y):
        #         x[0][0][0] =  y[0][0][1]
        #         return torch.sum(x, dim=1)
        model_path = self.model_path
        config = self.config

        class WrapModel(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.vision_model = Qwen2VisionTransformer(config)

                def get_weights_iterator(model):
                    import glob

                    def prepare_weights(model):
                        allow_patterns = ["*.safetensors", "*.bin"]
                        hf_weights_files = []
                        for pattern in allow_patterns:
                            hf_weights_files += glob.glob(os.path.join(model, pattern))
                            if len(hf_weights_files) > 0:
                                if pattern == "*.safetensors":
                                    use_safetensors = True
                                break
                        return model, hf_weights_files, use_safetensors

                    hf_folder, hf_weights_files, use_safetensors = prepare_weights(
                        model
                    )
                    if use_safetensors:

                        def safetensors_weights_iterator(hf_weights_files):
                            """Iterate over the weights in the model safetensor files."""
                            enable_tqdm = (
                                not torch.distributed.is_initialized()
                                or torch.distributed.get_rank() == 0
                            )
                            from tqdm import tqdm

                            for st_file in tqdm(
                                hf_weights_files,
                                desc="Loading safetensors checkpoint shards",
                                disable=not enable_tqdm,
                                # bar_format=_BAR_FORMAT,
                            ):
                                from safetensors.torch import safe_open

                                with safe_open(st_file, framework="pt") as f:
                                    for name in f.keys():  # noqa: SIM118
                                        param = f.get_tensor(name)
                                        yield name, param

                        weights_iterator = safetensors_weights_iterator(
                            hf_weights_files
                        )
                    else:
                        raise ValueError
                    return weights_iterator

                self.vision_model.load_weights(
                    get_weights_iterator(model_path),
                )
                self.vision_model.float()

            def forward(
                self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, batch_tensor
            ) -> torch.Tensor:
                return self.vision_model(
                    hidden_states.squeeze(0), grid_thw.squeeze(0), batch_tensor
                ).unsqueeze(0)

        with torch.no_grad():
            simple_module = WrapModel().to("cuda")
            simple_module.eval()
            # test_images = torch.randn(1, 8064, 1176, dtype=torch.float32).to(device)
            # test_grid_thw = torch.tensor([[1, 84, 96]], dtype=torch.int32).unsqueeze(0).to(device)
            # print(simple_module(test_images, test_grid_thw)); exit(0)
            # simple_module(images, grid_thw, batch_tensor)
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            images = torch.randn(1, 5084, self.input_embed_dim, dtype=torch.float32).to(
                device
            )
            grid_thw = (
                torch.tensor([[1, 82, 62]], dtype=torch.int64).unsqueeze(0).to(device)
            )
            first_grid = grid_thw[0, 0, 0].item()
            batch_tensor = torch.zeros(first_grid).to(device)
            torch.onnx.export(
                simple_module,
                (images, grid_thw, batch_tensor),
                onnx_file_path,
                opset_version=17,
                input_names=["input", "grid_thw", "batch_tensor"],
                output_names=["output"],
                dynamic_axes={
                    "input": {
                        1: "nframes",
                    },
                    "grid_thw": {
                        1: "batch",
                    },
                },
            )
            # # # Check onnx
            # onnx.checker.check_model(model)
            # infer_model = onnx.shape_inference.infer_shapes(model)
            # onnx_output_file = onnx_file_path.split(".onnx")[0] + "_w_shapes.onnx"
            # onnx.save(infer_model, onnx_output_file)

        # release_gc()  # Further release memory
        print(
            f"Export to ONNX file successfully! The ONNX file stays in {onnx_file_path}"
        )

    def generate_trt_engine(self, onnxFile, planFile):
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)

        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
                print("Failed parsing %s" % onnxFile)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnxFile)

        profile.set_shape(
            "input",
            min=[1, 1, self.input_embed_dim],
            opt=[1, 10000, self.input_embed_dim],
            max=[1, 40000, self.input_embed_dim],
        )
        profile.set_shape("grid_thw", min=[1, 1, 3], opt=[1, 2, 3], max=[1, 4, 3])

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        if engineString is None:
            raise RuntimeError("Failed building %s" % planFile)
        else:
            print("Succeeded building %s in %d s" % (planFile, t1 - t0))
            with open(planFile, "wb") as f:
                f.write(engineString)


def get_engine_name(rank):
    return "rank{}.engine".format(rank)


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


@contextlib.contextmanager
def _scoped_stream():
    """Create a scoped cuda stream, and synchronize it when the context is destroyed"""
    # TODO: delete torch, use cuda native python bindings
    import torch

    stream = torch.cuda.current_stream()
    try:
        # return a handle, trt and other lib does not recognize torch.cuda.Stream
        yield stream.cuda_stream
    finally:
        stream.synchronize()


@dataclass
class TensorInfo:
    name: str
    dtype: trt.DataType
    shape: tuple
    # add more info like strides, formats if needed


class Session(object):
    """Session is a managed TensorRT runtime."""

    def __init__(self, **kwargs):
        # use Session.from_serialized_engine to create a session
        pass

    def _init(self, engine_buffer=None):
        """
        @brief: Setup TensorRT engines and context from a serialized engine file
        @param engine_buffer: a buffer holds the serialized TRT engine
        """
        logger = trt.Logger(trt.Logger.INFO)
        self._runtime = trt.Runtime(logger)
        if engine_buffer is not None:
            self._engine = self.runtime.deserialize_cuda_engine(engine_buffer)

        self._context = None
        if not self.engine.streamable_weights_size:
            self.__prepare_execution_contexts()
        return self

    def __prepare_execution_contexts(self):
        self._context = self.engine.create_execution_context()
        assert self._context is not None, "Failed to create an execution context!"
        with _scoped_stream() as stream:
            self._context.set_optimization_profile_async(0, stream)

    @staticmethod
    def from_serialized_engine(engine):
        """
        @brief: Create a session from a serialized engine
        @param engine: a serialized engine
        @return: a Session object
        """
        session = Session()
        return session._init(engine)

    @staticmethod
    def from_engine(engine):
        """
        @brief: Create a session from an existing ICudaEngine engine
        @param engine: an ICudaEngine
        @return: a Session object
        """
        session = Session()
        session.engine = engine
        return session._init()

    @property
    def runtime(self) -> trt.Runtime:
        return self._runtime

    @property
    def engine(self) -> trt.ICudaEngine:
        return self._engine

    @engine.setter
    def engine(self, engine: trt.ICudaEngine):
        self._engine = engine

    @property
    def context(self) -> trt.IExecutionContext:
        """
        @brief: Get the default TensorRT execution context,
            use self.engine.create_execution_context() to create a new context if needed
        @return: one TensorRT execution context object
        """
        return self._context

    @property
    def context_mem_size(self) -> int:
        return self.engine.device_memory_size_v2

    def _print_engine_info(self):
        """print engine info for debug purpose, internal use only."""
        refittable = self.engine.refittable
        num_layers = self.engine.num_layers
        device_memory_size = self.engine.device_memory_size_v2
        name = self.engine.name
        nb_profiles = self.engine.num_optimization_profiles
        print(
            f"Engine:{name=:}, {refittable=:}, {num_layers=:}, {device_memory_size=:}, {nb_profiles=:}"
        )
        self._print_io_info()

    def _print_io_info(self):
        """print engine i/o info for debug purpose, internal use only."""

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            tformat = ";".join(
                [
                    self.engine.get_tensor_format_desc(name, p)
                    for p in range(self.engine.num_optimization_profiles)
                ]
            )
            print(f"Tensor:{name=:}, {mode=:}, {shape=:}, {dtype=:}, {tformat=:}")

    def infer_shapes(
        self, inputs: List[TensorInfo], context: Optional[trt.IExecutionContext] = None
    ) -> List[TensorInfo]:
        """
        @brief: Set input shapes to given context, and infer the output shapes from the given input shapes.
               This function should be called every time when the input shapes are changed before calling run().
               Or call the context.set_input_shape on all dynamic shaped input tensors manually.
        @param inputs: list of TensorInfo object, each item represents an input tensor
        @param context: TensorRT execution context, if None, use the default context
        @return: list of TensorInfo object, each item represents an output tensor, returns None if failed
        """
        # set shape to the default context if context is not specified
        if context is None:
            context = self.context
        for i in inputs:
            if self.engine.get_tensor_mode(i.name) != trt.TensorIOMode.INPUT:
                raise ValueError(f"Tensor:{i.name} is not an input tensor")
            if self.engine.get_tensor_dtype(i.name) != i.dtype:
                raise ValueError(f"Tensor:{i.name} has wrong dtype")
            if not context.set_input_shape(i.name, i.shape):
                raise RuntimeError(
                    f"Could not set shape {i.shape} for tensor {i.name}. Please check the profile range for which your model was build."
                )

        outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = context.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                outputs.append(TensorInfo(name, dtype, shape))
        return outputs

    def _set_weight_streaming(self, gpu_weights_percent):
        if not self.engine.streamable_weights_size:
            assert (
                gpu_weights_percent == 1
            ), "Engine built without weight streaming. Cannot set gpu_weights_percent to a value other than 1."
            return

        assert self.engine is not None

        self._context = None

        min = 0
        max = self.engine.streamable_weights_size
        budget = int(gpu_weights_percent * max)

        self.engine.weight_streaming_budget_v2 = budget
        assert (
            self.engine.weight_streaming_budget_v2 == budget
        ), "Failed to set weight streaming budget!"
        print(
            f"Set gpu weights percent to {gpu_weights_percent}, which is {budget} bytes. Valid range: {min} bytes ~ {max} bytes."
        )

        try:
            self.__prepare_execution_contexts()
        except:
            free_mem = torch.cuda.mem_get_info()[0]
            if free_mem < budget:
                raise torch.cuda.OutOfMemoryError(
                    f"Out of Memory: Memory budget is {budget} bytes but only {free_mem} bytes are available on the GPU."
                )
            raise

    def run(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any], stream, context=None
    ) -> bool:
        """
        @brief: Run the TensorRT engine with the given inputs and outputs
        @param inputs: dict of input tensors, key is tensor name, value is tensor pointer or torch tensor
        @param outputs: dict of output tensors, key is tensor name, value is tensor pointer or torch tensor
        @param stream: cuda stream to enqueue the TensorRT engine on
        @param context: TensorRT execution context, if None, use the default context
        @return: True if enqueue succeeded, note the enqueue is an async call,
            returning True does not mean the execution is finished
        """
        # enqueue to the default context if context is not specified
        if context is None:
            context = self.context

        import torch

        for tensor_name in inputs:
            tensor = inputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor, torch.Tensor) else tensor
            context.set_tensor_address(tensor_name, ptr)
        for tensor_name in outputs:
            tensor = outputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor, torch.Tensor) else tensor
            context.set_tensor_address(tensor_name, ptr)
        ok = context.execute_async_v3(stream)
        return ok
