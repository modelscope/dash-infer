'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    client.py
'''
from ._allspark_client import AsClientEngine, AsStatus
from .engine_utils import EngineUtils

class ClientEngine(AsClientEngine):
    """
    class for allspark client engine which lauch mpi daemon services to process multinuma inferer.
    In this way, Allspark inferer served as daemon processes based on config numa nums, Client engine
    broadcasts request to Allspark daemon services and receive response based on grpc.
    Normally, it is used for CPU multi-numa inferer.
    """

    def __init__(self):
        super().__init__()
        self.engine = EngineUtils()
        self.engine.version = self.get_version_full()

    def serialize_model_from_torch(
            self,
            model_name,
            model_type,
            torch_model,
            model_config,
            save_dir,
            as_model_path = None,
            as_weight_path = None,
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
        return self.engine.dump_build_meta_to_proto(model_proto, weights_path,
                                                    torch_build_param)

    def build_model_from_config_struct(self, as_model_config_struct):
        self._build_model_from_as_model_config(as_model_config_struct)

    def start_model(
            self,
            model_name: str,
    ) -> AsStatus:
        return self._start_model(model_name)

    def stop_model(
            self,
            model_name: str,
    ) -> AsStatus:
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
        return self._start_request(model_name, inputs, generate_config)

    def get_no_wait(self, model_name: str, result_queue):
        return self._get_no_wait(model_name, result_queue)

    def get_wait(self, model_name: str, result_queue):
        return self._get_wait(model_name, result_queue)

    def get_output_no_wait(self, model_name: str, result_queue):
        return self._get_output_no_wait(model_name, result_queue)

    def get_output_wait(self, model_name: str, result_queue):
        return self._get_output_wait(model_name, result_queue)

    def get_request_status(self, model_name: str, result_queue):
        return self._get_request_status(model_name, result_queue)

    def stop_request(self, model_name: str, request_handle) -> AsStatus:
        return self._stop_request(model_name, request_handle)

    def release_request(self, model_name: str, request_handle) -> AsStatus:
        return self._release_request(model_name, request_handle)

    def sync_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Block all API calls to this model until the target request is complete.
        If request_handle is None, synchronize all requests.
        """
        return self._sync_request(model_name, request_handle)

    def get_free_frame(
            self,
            model_name: str,
    ) -> int:
        return self._get_free_frame(model_name)

    def get_as_engine_stat(
            self,
            model_name: str,
    ):
        return self._get_as_engine_stat(model_name)
