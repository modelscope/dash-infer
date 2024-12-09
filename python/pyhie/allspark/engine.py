'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    engine.py
'''
import os
from enum import Enum

import torch
from torch import Tensor

from ._allspark import AsStatus, AsEngine, AsEngineStat, AsModelConfig, AsCacheMode
from .engine_utils import EngineUtils
from .generation_config import ASGenerationConfigBuilder
from .model_loader import LLM


class RequestStatInfo():
    def __init__(self, info_dict):
        super().__init__()
        try:
            self.arrival_time = info_dict[RequestStatInfo.Key.KEY_REQUEST_TIME_TS.value]
            self.first_token_time = info_dict[RequestStatInfo.Key.KEY_FIRST_TOKEN_TIME_TS.value]
            self.last_token_time = info_dict[RequestStatInfo.Key.KEY_LAST_TOKEN_TIME_TS.value]
            self.time_to_first_token = info_dict[RequestStatInfo.Key.KEY_TTFT_MS.value]
            self.time_in_queue = info_dict[RequestStatInfo.Key.KEY_SCHEDULE_TIME_MS.value]
            self.finish_time = info_dict[RequestStatInfo.Key.KEY_TOTAL_TIME_MS.value]
            self.context_token_length = info_dict[RequestStatInfo.Key.KEY_INPUT_LEN.value]
            self.generated_token_length = info_dict[RequestStatInfo.Key.KEY_OUTPUT_LEN.value]
            self.context_cached_length = info_dict[RequestStatInfo.Key.KEY_INPUT_CACHE_LEN.value]
            self.context_tps = info_dict[RequestStatInfo.Key.KEY_CONTEXT_TPS.value]
            self.generate_tps = info_dict[RequestStatInfo.Key.KEY_GENERATE_TPS.value]
        except KeyError as e:
            print("key not found, stat info may incomplete.")

    def __str__(self):
        return (f"RequestStatInfo("
                f"arrival_time={self.arrival_time}, "
                f"first_token_time={self.first_token_time}, "
                f"last_token_time={self.last_token_time}, "
                f"time_to_first_token={self.time_to_first_token}, "
                f"time_in_queue={self.time_in_queue}, "
                f"finish_time={self.finish_time}, "
                f"context_token_length={self.context_token_length}, "
                f"generated_token_length={self.generated_token_length}, "
                f"context_cached_length={self.context_cached_length}, "
                f"context_tps={self.context_tps}, "
                f"generate_tps={self.generate_tps})")

    class Key(Enum):
        KEY_REQUEST_TIME_TS = "arrival_time"
        KEY_FIRST_TOKEN_TIME_TS = "first_token_time"
        KEY_LAST_TOKEN_TIME_TS = "last_token_time"
        KEY_TTFT_MS = "time_to_first_token"
        KEY_SCHEDULE_TIME_MS = "time_in_queue"
        KEY_TOTAL_TIME_MS = "finished_time"
        KEY_INPUT_LEN = "context_token_length"
        KEY_OUTPUT_LEN = "generated_token_length"
        KEY_INPUT_CACHE_LEN = "context_cached_length"
        KEY_CONTEXT_TPS = "context_tps"
        KEY_GENERATE_TPS = "generate_tps"


# TODO: 1) add lora  2) add iq quant.
class TargetDevice(Enum):
    """ target inference device for runtime Config"""
    CUDA = "CUDA",
    """CUDA-like device """
    CPU = "CPU",
    """ CPU device without NUMA-aware """
    CPU_NUMA = "CPU_NUMA"
    """ CPU Device with NUMA-aware """


class RoPEScaleMethod(Enum):
    """
    Enum type of RoPE Scaling method.

    YaRN: https://github.com/jquesnelle/yarn method
    """
    NO_SCALE = 0
    YaRN = 3
    NTK = 4


class Engine(AsEngine):
    from .model_loader import HuggingFaceModel

    def __init__(self):
        super().__init__()
        self.engine = EngineUtils()
        self.engine.version = self.get_version_full()
        self.sent_vocab = False

    def __repr__(self):
        return f"AsEngine: version={self.get_version_full()} "

    def serialize_model_from_torch(
            self,
            model_name,
            model_type,
            torch_model,
            model_config,
            save_dir=None,
            as_model_path=None,
            as_weight_path=None,
            data_type="float32",
            derive_type="lmhead",
            multigpu_mode=1,
            do_binary_add_fused=True,
            do_dynamic_quantize_convert=False,
            quant_config=None,
            enable_i8cache_mha=False,
            mha_kv_cache_config=None,
            # dump_param_to_dict = {},
            use_dynamic_ntk=False,
            use_logn_attn=False,
            model_sequence_length=2048,
            seqlen_extrapolation=1.0,
            lora_cfg=None,
            rotary_base=10000.0,
    ):
        """
        Convert a pytorch model(huggingface format or a megatron format) to allspark model format,
        the weight process, including quantize will be process in this phase.
        Args:
            model_name: the model file name
            model_type: the model tag in allspark
            torch_model: the pytorch model path
            model_config: the model's model config in allspark format
            save_dir: the converted model store path.
            data_type: the data type of this model, default is 'float32', candidate will be [float16, bfloat16, float32]
            derive_type: [TODO] move to model config
            multigpu_mode: [TODO] multiple mode, remove
            do_binary_add_fused: [TODO] move to RuntimeConfig
            do_dynamic_quantize_convert: [TODO] move to Runtime Config,
            quant_config:  [TODO] QuantConfig class
            enable_i8cache_mha: [TODO] remove
            mha_kv_cache_config: [TODO] remove
            use_dynamic_ntk:  [TODO] move to model config
            use_logn_attn:  [TODO] move to model config
            model_sequence_length: [TODO] move to model config.
            seqlen_extrapolation:  [TODO] move to model config
            lora_cfg:   [TODO] LORA config
            rotary_base: [TODO] move to model config.

        Returns: This function don't return, will throw exception if meets error.

        """
        print("rotary base: ", rotary_base)

        return self.engine.serialize_model_from_torch(
            model_name,
            model_type,
            torch_model,
            model_config,
            save_dir,
            as_model_path,
            as_weight_path,
            data_type,
            derive_type,
            multigpu_mode,
            do_binary_add_fused,
            do_dynamic_quantize_convert,
            quant_config,
            enable_i8cache_mha,
            mha_kv_cache_config,
            # dump_param_to_dict,
            use_dynamic_ntk,
            use_logn_attn,
            model_sequence_length,
            seqlen_extrapolation,
            lora_cfg,
            rotary_base)

    def dump_build_meta_to_proto(self, model_proto, weights_path,
                                 torch_build_param):
        """Helper function dump the model file into torch param, debug api. """
        return self.engine.dump_build_meta_to_proto(model_proto, weights_path,
                                                    torch_build_param)

    def install_model(self, as_model_config_struct: AsModelConfig):
        """
            Builds a model from the provided configuration.

            This will install the model into engine,
            and `as_model_config_struct.model_name` can be reference in other api.

            Args:
                as_model_config_struct (AsModelRuntimeConfig): A reference to the model configuration object,
                                                    containing all necessary settings for building the model.

            Returns:
                AsStatus: An AsStatus value indicating the success or failure of the model
                          building operation. Returns AS_SUCCESS if successful; otherwise,
                          returns an error code specifying the encountered issue.
            """
        self._build_model_from_as_model_config(as_model_config_struct)

    def build_model_from_config_struct(self, as_model_config_struct: AsModelConfig):
        """
            Builds a model from the provided configuration.
            deprecated function: call install_model

            This will install the model into engine,
            and `as_model_config_struct.model_name` can be reference in other api.

            Args:
                as_model_config_struct (AsModelRuntimeConfig): A reference to the model configuration object,
                                                    containing all necessary settings for building the model.

            Returns:
                AsStatus: An AsStatus value indicating the success or failure of the model
                          building operation. Returns AS_SUCCESS if successful; otherwise,
                          returns an error code specifying the encountered issue.
            """
        self._build_model_from_as_model_config(as_model_config_struct)

    def process_json_mode_arguments(self, generate_config: dict):
        if self.sent_vocab == False:
            response_format = generate_config.get("response_format", None)
            if response_format != None:
                # user put response_format dict in gen_cfg
                type = response_format.get("type", None)
                if type == None or type != "json_object":
                    generate_config.pop("vocab", None)
                    generate_config.pop("vocab_type", None)
                    print('''[Warning]AsEngine: no valid \'type\' in \'response_format\', guided decoding will not be effective''')
                else:
                    # check if there's valid vocab
                    vocab = generate_config.get("vocab", None)
                    if vocab == None:
                        print('''[Warning]AsEngine: provided \'response_format\' when there's no vocab, guided decoding will not be effective''')
                    else:
                        self.sent_vocab = True
            else:
                # no json schema str provided, remove vocab and vocab_type in gen_cfg
                generate_config.pop("vocab", None)
                generate_config.pop("vocab_type", None)
        else:
            # vocab already sent to the engine by a previous request, not needed anymore
            generate_config.pop("vocab", None)
            generate_config.pop("vocab_type", None)

    def start_model(
            self,
            model_name: str,
    ) -> AsStatus:
        """
        Start running this model, model start to spinning.
        Args:
            model_name: model name

        Returns:
        AsStatus Code, AsSuccess
        """
        return self._start_model(model_name)

    def stop_model(
            self,
            model_name: str,
    ) -> AsStatus:
        """
        Stop the model running, the output queue will be return in a stopped status.
        Args:
            model_name: installed model name.

        Returns: ASSatus: the api status, ASSuccess means success, otherwise means error.

        """
        return self._stop_model(model_name)

    def release_model(
            self,
            model_name: str
    ) -> AsStatus:
        return self._release_model(model_name)

    def start_request(
            self,
            model_name: str,
            inputs,
            generate_config={},
    ):
        """
            Initiate a generation request with a model and general inputs，
            This function input is an input dict, you can use `start_request_ids` and  `start_request_text`
            for a simpler usage.

            Args:
                model_name (str): The name of the installed model to use for generation.
                inputs (dict): The input data for the generation process, input format should be like this:
                ```
                input_dict = {
                    "input_ids": torch.utils.dlpack.to_dlpack(input_ids.cpu()),
                }
                ```
                generate_config (dict, optional): Configuration dictionary for generation parameters

            Returns:
                tuple: A tuple containing three elements:
                    - ASStatus: Status code indicating the result of the request initiation.
                    - object: A request handle to identify this request in subsequent API calls.
                    - ResultQueue: A queue to fetch the outputs and status of this generation request.
            """
        # For JSON Mode
        self.process_json_mode_arguments(generate_config)
        return self._start_request(model_name, inputs, generate_config)

    def start_request_ids(self,
                          model_name: str,
                          model: LLM,
                          input_ids: Tensor,
                          generate_config_builder: ASGenerationConfigBuilder):
        """
            Start a generation request with a model and tensor inputs along with a structured generation configuration.

            Args:
                model_name (str): The name of the model installed for text generation tasks.
                model (LLM): the model
                input_ids (Tensor): Tensor containing the input token IDs for generation.
                generate_config_builder: generate config builder class


            Returns:
                tuple: A tuple consisting of:
                    - ASStatus: The status of the request as returned by the engine.
                    - object: A request handle to track and manage this specific request.
                    - ResultQueue: A queue from which to retrieve the results and status updates of the generation process
        """

        def is_one_dimensional(lst):
            return all(not isinstance(element, list) for element in lst)

        input_dict: dict = {"input_ids": None}
        import torch.utils.dlpack as dlpack
        if type(input_ids) == type(torch.LongTensor):
            if input_ids.dim() > 1:
                raise ValueError("input ids must 1 dim.")
            input_dict.update({
                "input_ids":
                    dlpack.to_dlpack([input_ids.cpu()]),  # input is 2-dim, wrap with []
            })
        elif type(input_ids) == type(torch.Tensor):
            if input_ids.dim() > 1:
                raise ValueError("input ids must 1 dim.")
            input_dict.update({
                "input_ids":
                    dlpack.to_dlpack([torch.LongTensor(input_ids.cpu())]),  # input is 2-dim, wrap with [], long
            })
        elif isinstance(input_ids, list):
            if not is_one_dimensional(input_ids):
                raise ValueError("input ids must 1 dim.")

            input_dict.update({
                "input_ids":
                    dlpack.to_dlpack(torch.LongTensor([input_ids]).cpu()),  # input is 2-dim, wrap with []
            })
        else:
            raise ValueError(
                f"input ids type must be torch.Tensor, torch.LongTensor, or python list, give: {type(input_ids)} {isinstance(input_ids, list)}")

        gen_config = generate_config_builder.build()

        print(f"Start Request with Generate Config:")
        self.print_gen_cfg(gen_config)
        # For JSON Mode
        self.process_json_mode_arguments(gen_config)

        status, request_handle, result_queue = self._start_request(
            model_name, input_dict, gen_config)

        return status, request_handle, result_queue
        # construct a

    def print_gen_cfg(self, gen_cfg):
        fields_to_exclude = ['vocab']
        new_dict = {k: v for k, v in gen_cfg.items() if k not in fields_to_exclude}
        print(new_dict)
    def start_request_text(self,
                           model_name: str,
                           model: LLM,
                           input_str_or_array,
                           generate_config_builder: ASGenerationConfigBuilder):
        """
        Start Request by model and with text input.

        This function will call the tokenizer can be found by the model loader,
        and convert into input format required by engine in generate task.

        Args:
            model_name (str): installed model name.
            model (LLM): the model
            input_str_or_array (str, List[str]): input context pass
            generate_config_builder: generate config builder class

        Returns: A tuple consisting of:
                    - ASStatus: The status of the request as returned by the engine.
                    - object: A request handle to track and manage this specific request.
                    - ResultQueue: A queue from which to retrieve the results and status updates of the generation process.
        """
        tokenizer = model.init_tokenizer().get_tokenizer()
        input_ids = tokenizer.encode(input_str_or_array)
        input_dict = {
            "input_ids":
                torch.utils.dlpack.to_dlpack(Tensor([input_ids]).to(torch.int64).cpu()),
        }

        gen_config_dict = generate_config_builder.build()
        print("Start Request with Generate Config:")
        self.print_gen_cfg(gen_config_dict)
        # For JSON Mode
        self.process_json_mode_arguments(gen_config_dict)
        status, request_handle, result_queue = self._start_request(model_name, input_dict, gen_config_dict)

        return status, request_handle, result_queue

    def get_no_wait(self, model_name: str, result_queue):
        """
        (deprecated API) Retrieves a result from the queue without waiting.


        This API is deprecated, please use function  `ResultQueue.GetNoWait()`

        Returns:
           GeneratedElements: Smart pointer to the generated elements.
        """
        return self._get_no_wait(model_name, result_queue)

    def get_wait(self, model_name: str, result_queue):
        """
          (deprecated API) Fetches a result from the queue, blocking until a new token is generated.


          This API is deprecated, please use function  `ResultQueue.Get()`

          Returns:
             GeneratedElements: Smart pointer to the generated elements.
          """
        return self._get_wait(model_name, result_queue)

    def get_request_status(self, model_name: str, result_queue):
        """
        Generate status of a request.

        This API is deprecated, please use function  `ResultQueue.GenerateStatus()`


        Returns:
            GenerateRequestStatus: The status of the generation request.
        """
        return self._get_request_status(model_name, result_queue)

    def stop_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Stops a request.

        Args:
            model_name (str): Model name.
            request_handle: Handle for the request.

        Returns:
            AsStatus: Status of the operation.
        """
        return self._stop_request(model_name, request_handle)

    def release_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Releases a request's resources.

        Args:
            model_name (str): Model name.
            request_handle: Handle for the request.

        Returns:
            AsStatus: Status of the operation.
        """
        return self._release_request(model_name, request_handle)

    def sync_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Waits for the completion of an asynchronous request.

        Args:
            model_name (str): Model name.
            request_handle: Handle for the request.

        Returns:
            AsStatus: Status of the operation.
        """

        return self._sync_request(model_name, request_handle)
