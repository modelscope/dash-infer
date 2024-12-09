import torch
import torch.nn as nn
import nvtx
from typing import Any, Dict, List, Optional
import tensorrt as trt
import contextlib
from dataclasses import dataclass

logger = trt.Logger(trt.Logger.WARNING)


class HieModel_V2(nn.Module):
    def __init__(self, model_fn="", input=dict()):
        super(HieModel_V2, self).__init__()
        # self.device_type = hie.DeviceType.CPU if device.type == "cpu" else hie.DeviceType.GPU
        # self.device_id = 0 if device.type == "cpu" else device.index
        # graphopt = hie.CreateGraphOPT()
        # graphopt.SetTargetDevice(self.device_type, self.device_id)
        # graphopt.AddInput(input)
        # if model_fn.endswith(".hie"):
        #     model_type = hie.ModelType.FRAMEWORK_HIE
        # else:
        #     model_type = hie.ModelType.FRAMEWORK_ONNX
        # model = graphopt.BuildModel(model_fn, model_type)
        # self.inferer = hie.CreateInferer(self.device_type, self.device_id, model)
        # self.ctx = self.inferer.CreateExecutionContext()

    def freeze(self):
        # self.inferer.FreezeWeight(0)
        pass

    def unfreeze(self):
        # self.inferer.UnFreezeWeight(0)
        pass

    def clean_cache(self):
        # self.ctx.DestoryIntermediateBuffer()
        pass

    def restore_cache(self):
        # self.ctx.RestoreIntermediateBuffer()
        pass


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

    def set_shapes(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        context: Optional[trt.IExecutionContext] = None,
    ):
        if context is None:
            context = self.context

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                ok = context.set_input_shape(name, tensor_dict[name].shape)
                logger.debug(
                    f"setting input tensor {name} with shape {tensor_dict[name].shape}"
                )
                if not ok:
                    raise ValueError(
                        f"Couldn't assign {name} with shape {tensor_dict[name].shape}, "
                        f"engine supports [min, opt, max] = {self.engine.get_tensor_profile_shape(name, context.active_optimization_profile)}"
                    )

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


class VisualTRT_V2(HieModel_V2):

    def __init__(self, vit_engine_path, trt_vit_config, input=input):
        print("loading qwen2-vit by pyhie")
        self.stream = torch.cuda.current_stream().cuda_stream
        print(f"Loading engine from {vit_engine_path}")
        with open(vit_engine_path, "rb") as f:
            engine_buffer = f.read()
        print(f"Creating session from engine {vit_engine_path}")
        self.session_vit = Session.from_serialized_engine(engine_buffer)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.trt_vit_config = trt_vit_config

        super(VisualTRT_V2, self).__init__(vit_engine_path, input=input)

    @nvtx.annotate()
    def forward(self, images, grid_thw, batch, use_flashattn=True):
        images = images.unsqueeze(0)
        grid_thw = grid_thw.unsqueeze(0)
        visual_inputs = {
            "input": images.to(torch.float32),
            "grid_thw": grid_thw.to(torch.int64),
        }
        # visual_output_info = self.session_vit.infer_shapes(
        #     [TensorInfo("input", trt.DataType.FLOAT, images.shape), TensorInfo("grid_thw", trt.DataType.INT64, grid_thw.shape)])
        # visual_outputs = {
        #     t.name: torch.empty(tuple(t.shape),
        #                         dtype=trt_dtype_to_torch(t.dtype),
        #                         device="cuda")
        #     for t in visual_output_info
        # }
        self.session_vit.context.set_input_shape("input", images.shape)
        self.session_vit.context.set_input_shape("grid_thw", grid_thw.shape)
        hidden_size = self.trt_vit_config.hidden_size
        embed_dim = self.trt_vit_config.embed_dim
        spatial_merge_size = self.trt_vit_config.spatial_merge_size
        image_tokens = int(
            visual_inputs["input"].shape[1]
            * embed_dim
            / (embed_dim * (spatial_merge_size**2))
        )
        visual_outputs = {
            "output": torch.empty(
                (1, image_tokens, hidden_size), dtype=torch.float32, device="cuda"
            )
        }
        # profiler.start("ViT")
        ok = self.session_vit.run(visual_inputs, visual_outputs, self.stream)
        # profiler.stop("ViT")
        # Vit_time = profiler.elapsed_time_in_sec("ViT")
        # print(f"TensorRT-LLM ViT latency: {Vit_time:3f} sec ")
        assert ok, "Runtime execution failed for vit session"

        return visual_outputs["output"].squeeze(0).clone()
