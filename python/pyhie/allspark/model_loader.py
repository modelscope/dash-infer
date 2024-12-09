'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    model_loader.py
'''
import glob
import os
import sys

import torch
import torch.cuda
import enum

from safetensors.torch import safe_open
from typing import Any, Dict, Generator, List, Tuple
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig

from .generation_config import ASGenerationConfigBuilder
from .model_config import ModelAdapterFactory
from ._allspark import AsModelConfig
from .quant.gptq_iq_adapter import GPTQ2IQWeightAdapter
from .quantization import QuantizeConfig
from ._allspark import VocabType


class ConfigFieldMissingError(ValueError):
    def __init__(self, field, config_dict):
        super().__init__(f"config required field '{field}'  is missing, full content: {config_dict}")


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"


def _filter_files_not_needed_for_inference(hf_weights_files: List[str]) -> List[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


def _prepare_weight(
        model_name_or_path: str,
        fall_back_to_pt: bool,
        load_format: str,
) -> Tuple[str, List[str], bool]:
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == LoadFormat.AUTO:
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == LoadFormat.SAFETENSORS:
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == LoadFormat.PT:
        allow_patterns = ["*.pt"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        print(f"model path '{model_name_or_path}' doesn't exist")
    else:
        hf_folder = model_name_or_path

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break

    if not use_safetensors:
        hf_weights_files = _filter_files_not_needed_for_inference(hf_weights_files)

    if len(hf_weights_files) == 0:
        raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


def _load_safetensors_weights(
        hf_weights_files: List[str],
) -> Dict[str, torch.Tensor]:
    """Iterate over the weights in the model safetensor files."""
    state_dict = dict()
    for st_file in hf_weights_files:
        with safe_open(st_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def _load_pt_weights(
        hf_weights_files: List[str],
) -> Dict[str, torch.Tensor]:
    """Iterate over the weights in the model bin/pt files."""
    state_dict = dict()
    for bin_file in hf_weights_files:
        state_dict.update(torch.load(bin_file, map_location="cpu"))
    return state_dict


def get_state_dict(
        model_name_or_path: str,
        fall_back_to_pt: bool,
        load_format: str,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Get an iterator for the model weights based on the load format."""
    hf_folder, hf_weights_files, use_safetensors = _prepare_weight(
        model_name_or_path, fall_back_to_pt, load_format
    )
    if use_safetensors:
        return _load_safetensors_weights(hf_weights_files)
    return _load_pt_weights(hf_weights_files)


class LLM:
    def load_model(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def init_quantization(self, enable=True, customized_quant_config = None):
        raise NotImplementedError("Subclasses must implement this method")

    def init_tokenizer(self, padding_side="left"):
        raise NotImplementedError("Subclasses must implement this method")

    def get_tokenizer(self) -> AutoTokenizer:
        raise NotImplementedError("Subclasses must implement this method")

    def read_model_config(self):
        raise NotImplementedError("Subclasses must implement this method")

    def create_reference_generation_config_builder(self, runtimeConfig: AsModelConfig):
        raise NotImplementedError("Subclasses must implement this method")

    def get_model_context_length(self):
        raise NotImplementedError("Subclasses must implement this method")

    def create_reference_runtime_config_builder(self, engine_model_name, device_type, device_ids, max_batch,
                                                max_context_length=-1) -> "AsModelRuntimeConfigBuilder":
        raise NotImplementedError("Subclasses must implement this method")

    def free_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def serialize_to_memory(self, engine, enable_quant=False, weight_only_quant=False):
        raise NotImplementedError("Subclasses must implement this method")

    def free_memory_serialize_file(self):
        raise NotImplementedError("Subclasses must implement this method")

    def serialize_to_path(self, engine, model_output_dir: str):
        raise NotImplementedError("Subclasses must implement this method")

    def get_model_config(self):
        raise NotImplementedError("Subclasses must implement this method")


class DashInferModel(LLM):
    """
    This class can store and load config from a dashinfer(allspark) model,
    it will load dashinfer(allspark) format model and related config file
    """

    def _load_yaml_config(self, file_path, err_str):
        import yaml
        try:
            with open(file_path, 'r') as file:
                try:
                    self.di_model_config = yaml.safe_load(file)
                except yaml.YAMLError as yml_err:
                    raise ValueError("config parse error") from yml_err
        except FileNotFoundError as e:
            raise ValueError(
                err_str) from e

    def __init__(self, pretrain_model_name_or_path, pretrain_model_name="model", config_file_path=None,
                 config_content=None):
        """
        Args:
            pretrain_model_name_or_path: local model path
            pretrain_model_name:  model name, use this name as prefix find model's file, use "model" as default name
            config_file_path: model's config path, if this value is None, will fetch di_config.yaml file in model's folder.
            config_content: config content by string, will override file path.
        """
        super().__init__()

        import yaml
        self.dimodel_config_name = "dimodel_config.yaml"
        self.safe_model_name = pretrain_model_name.replace("/", "_")

        if config_content is None and config_content is None:
            # not provide config file and content, going to find the dimodel_config.yaml under model's folder.
            config_path = os.path.join(pretrain_model_name_or_path, self.dimodel_config_name)
            self._load_yaml_config(config_path, f"model config not found in model folder {config_path}, model file "
                                                f"and content also not provided.")

        else:
            if config_content is not None:
                print("config file content provided, will override model's folder's content")
                try:
                    self.di_model_config = yaml.safe_load(config_content)
                except yaml.YAMLError as yml_err:
                    raise ValueError("config parse error") from yml_err
            else:  # means config_file_path exists.
                print("config file content provided, will override model's folder's content")
                self._load_yaml_config(config_file_path, f"model config not found in path {config_file_path}.")

        self.di_model_path = pretrain_model_name_or_path

        self.fill_config_by_di_config()

    def fill_config_by_di_config(self):
        # check each part exits.

        from .runtime_config import AsModelRuntimeConfigBuilder

        if "generation_config" not in self.di_model_config:
            raise ConfigFieldMissingError("generation_config", self.di_model_config)
        if "runtime_config" not in self.di_model_config:
            raise ConfigFieldMissingError("runtime_config", self.di_model_config)
        if "model" not in self.di_model_config:
            raise ConfigFieldMissingError("model", self.di_model_config)

        self.base_generation_config = ASGenerationConfigBuilder(self.di_model_config['generation_config'])
        self.base_runtime_config_builder = AsModelRuntimeConfigBuilder()

        rfield = self.di_model_config['runtime_config']

        self.base_runtime_config_builder.from_dict(rfield)

        # setup cuda mem related setting.
        if "cuda_mem" in rfield:

            if "memory_ratio" in rfield["cuda_mem"]:
                os.putenv("BFC_MEM_RATIO", rfield["cuda_mem"]["memory_ratio"])
            if "reserve_size_mb" in rfield["cuda_mem"]:
                os.putenv("BFC_LEFTOVER_MB", rfield['cuda_mem']['reserve_size_mb'])

        # TOOD: setup the rope scaling

        # generation config.
        # diconfig use may have stop_word_ids

    def load_model(self):
        """
        load mode and config, but in DI Model, actual file read happens in
        Args:
            config_path_path:

        Returns:

        """
        return self

    def init_quantization(self, enable=True, customized_quant_config=None):
        pass

    def init_tokenizer(self, padding_side="left"):
        pass

    def get_tokenizer(self) -> AutoTokenizer:
        pass

    def read_model_config(self):
        pass

    def create_reference_generation_config_builder(self, runtimeConfig: AsModelConfig = None):
        as_generate_config = ASGenerationConfigBuilder(None)

        gen_cfg_field = self.di_model_config["generation_config"]

        as_generate_config.process_eos_tokens(gen_cfg_field['eos_token_id'], as_generate_config.dict_store)

        as_generate_config.update(self.di_model_config["generation_config"])

        if runtimeConfig != None:
            as_generate_config.update({"max_length": min(gen_cfg_field['max_length'], runtimeConfig.engine_max_length)})

        return as_generate_config

    def get_model_context_length(self):
        """
        this function only return model's original context length,
        some RoPE scaling method maybe affect the actual value, but since it's a runtime config.
        move the check into dynamic place.

        Returns: model's original context length
        """

        # TODO: use max position embedding ? or use runtime config max length ?
        if 'max_position_embeddings' in self.di_model_config['model']:
            return self.di_model_config['model']['max_position_embeddings']
            # this
        else:
            # unlimit this value if not setup correctly in model's config.
            return sys.maxsize

    def create_reference_runtime_config_builder(self) -> "AsModelRuntimeConfigBuilder":
        import copy
        return copy.deepcopy(self.base_runtime_config_builder)

    def free_model(self):
        return self

    def serialize_to_memory(self, engine, enable_quant=False, weight_only_quant=False):
        return self

    def free_memory_serialize_file(self):
        return self

    def serialize_to_path(self, engine, model_output_dir: str):
        return self

    def get_model_config(self):
        pass


class ModelSerializerException(Exception):
    """Model Serializer exception. """

    def __init__(self, message, code, model_name):
        super().__init__(message)
        self.model_name = model_name
        self.code = code

    def __str__(self):
        return f'[{self.code}] {self.model_name}'


class HuggingFaceModel(LLM):
    """
    This class can store load information for a huggingface model, easier to init model config and quantization etc.
    """

    def __init__(self, pretrain_model_name_or_path, pretrain_model_name,
                 in_memory_serialize=False,
                 user_set_data_type=None,
                 trust_remote_code=False):
        """
        Args:
            user_set_data_type: user set inference data type, choice [float32, bfloat16, float16]
            pretrain_model_name_or_path: local model path or pretrain model name
            pretrain_model_name: local model name used create serializer model file
            in_memory_serialize: store the serialize in memory file
            trust_remote_code: trust_remote_code pass to huggingface sdk, default value is 'False', can be [True,False]
            data_type: the model's data type, default value is 'bfloat16', can be [float32,bfloat16, float16]
        """
        self.temp_weight_file = None
        self.temp_model_file = None
        from . import QuantizationSettings

        super().__init__()
        self.user_set_data_type = None
        self.adapter = None
        self.quant_setting: QuantizationSettings = None
        self.hf_quant_config = None
        self.hf_model_config = None
        self.torch_model_state_dict = None
        self.as_model_config = None
        self.tokenizer = None
        self.torch_model = None
        self.hf_model_path = pretrain_model_name_or_path
        self.pretain_model_name = pretrain_model_name
        self.in_memory_serialize = in_memory_serialize
        # For JSON Mode
        self.vocab = None
        self.vocab_type = None

        # assume local file and model id, should same.
        if pretrain_model_name_or_path != pretrain_model_name:
            self.have_local_file = True
        else:
            self.have_local_file = False

        if user_set_data_type:
            self.user_set_data_type = user_set_data_type
            self.data_type = user_set_data_type
        else:
            self.data_type = "bfloat16"
        self.trust_remote_code = trust_remote_code
        # store path and parse check path, if it's a model hf model.

    def load_model(self, override_data_type=None, direct_load=False, load_format="auto", **kwargs):
        """
        Load a model and parse its weights,
        expecting the model to be in Huggingface's format.
        The model can either be downloaded locally using `snapshot_download`
        or be stored in a local file directory.
        This function loads the model into CPU memory and prepares it for conversion.

        Parameters:
        - model_path: The path to the model, which can be a remote URL or a local file path.
        - kargs: other key-val args will be forward to AutoModelForCausalLM.from_pretrained interface.
        Return self
        """
        kwargs["device_map"] = "cpu"
        # for model convert, only require cpu memory
        if not direct_load:
            # the open-source model can be loaded by huggingface 
            try:
                self.torch_model = AutoModelForCausalLM.from_pretrained(
                    self.hf_model_path, trust_remote_code=self.trust_remote_code,
                    **kwargs).eval()
                self.torch_model = self.torch_model.cpu()

                if self.data_type == 'float32':
                    self.torch_model.float()
                elif self.data_type == 'float16':
                    if self.torch_model.dtype != torch.float16:
                        self.torch_model.half()
                    self.torch_model.half()
                elif self.data_type == 'bfloat16':
                    if self.torch_model.dtype != torch.bfloat16:
                        self.torch_model.bfloat16()
                else:
                    self.torch_model = None
                    raise ValueError("unsupported data type: {}".format(
                        self.data_type))
            except Exception as e:
                print(f"exception when load model: {self.hf_model_path} , exception: {e}")
                raise ModelSerializerException("error load from huggingface", 1, self)

            # allspark only needs the torch's model's state dict, no needs to keep torch model
            self.torch_model_state_dict = self.torch_model.state_dict()
            self.torch_model = None
        else:
            # some interal models change the method to save weight of models, we direct load the weight
            try:
                self.torch_model_state_dict = get_state_dict(self.hf_model_path, fall_back_to_pt=False,
                                                             load_format="auto")
                dtype_dict = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
                if self.data_type in dtype_dict.keys():
                    convert_dtype = dtype_dict[self.data_type]
                    for k, v in self.torch_model_state_dict.items():
                        if v.dtype in dtype_dict.values():
                            self.torch_model_state_dict[k] = v.to(convert_dtype)
                else:
                    self.torch_model_state_dict = None
                    raise ValueError("unsupported data type: {}".format(
                        self.data_type))
            except:
                print(f"exception when load model: {self.hf_model_path} , exception: {e}")
                raise ModelSerializerException("error load from ", 1, self)

        # init the model config
        self.read_model_config()

        # For JSON Mode
        # try to get tokenizer
        try:
            self.init_tokenizer()
        except Exception as e:
            print('''[Warning]Model loader: failed to get tokenizer from model, JSON Mode disabled''')
            return self
        # try to get model vocab
        try:
            self.vocab = self.tokenizer.get_vocab()
        except Exception as e:
            print('''[Warning]Model loader: failed to get model vocab, JSON Mode disabled''')
            return self
        # try to get tokenizer type from 'tokenizer.json', if fail, use default type
        import json
        tokenizer_file = os.path.join(self.hf_model_path, 'tokenizer.json')
        self.vocab_type = VocabType.VOCAB_TYPE_BPE
        try:
            with open(tokenizer_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            print(f'''[Warning]Model loader: failed to parse JSON file: '{tokenizer_file}' to get tokenizer type, using default 
                type(BPE), output may be abnormal if using JSON Mode''')
            return self
        if "model" in data and "type" in data["model"]:
            if data["model"]["type"] == "BPE":
                self.vocab_type = VocabType.VOCAB_TYPE_BPE
            elif data["model"]["type"] == "WPM":
                self.vocab_type = VocabType.VOCAB_TYPE_WPM
            elif data["model"]["type"] == "SPM":
                self.vocab_type = VocabType.VOCAB_TYPE_SPM
            elif data["model"]["type"] == "UGM":
                self.vocab_type = VocabType.VOCAB_TYPE_UGM
            else:
                print(f'''[Warning]Model loader: model tokenizer type('{data["model"]["type"]}') not supported, JSON Mode disabled''')
                self.vocab = None
                self.vocab_type = None
        else:
            print('''[Warning]Model loader: key 'type' not found in model tokenizer.json, using default tokenizer 
                type(BPE), output may be abnormal if using JSON Mode''')
        # read quant config and tokenizer init should be done by user.
        return self

    def set_rope_scale_method(self, method: "RoPEScaleMethod", parameter_dict: dict):
        """
         setup rope scaling related parameter, call this function before call serialize related function, and after read_model_config()

        Args:
            method: Enum type of rope scale type.
            parameter_dict:  the rope_scaling dict in hf config, like
                            {
                            "factor": 4.0,
                            "original_max_position_embeddings": 32768,
                            "type": "yarn"
                            }

        Returns: self

        """

        self.as_model_config = self.adapter.get_as_model_config(rope_scale_method=method,
                                                                rope_scale_parameter=parameter_dict)
        return self

    def init_quantization(self, enable=True, customized_quant_config = None):
        from . import QuantizationSettings

        """
        load the quantization config from huggingface and enable in model inference.

        Args:
            enable: True or False, True means converted model will be quantization by model's config,
                    False means only load config for later use.
            customized_quant_config: Dict, customized quant config.

        Returns: self

        """
        if customized_quant_config:
            self.quant_setting = QuantizationSettings.from_hf_config(customized_quant_config)
            pass
        else:
            self.hf_quant_config = getattr(self.hf_model_config, "quantization_config", None)
            print(self.hf_model_config)
            if not self.hf_quant_config:
                raise ValueError("init quant failed, model config not include quantization config.")
            self.quant_setting = QuantizationSettings.from_hf_config(self.hf_quant_config)
        return self

    def init_tokenizer(self, padding_side="left"):
        """
        init model tokenizer, only init once.

        Args:
            padding_side: padding side pass to huggingface sdk.

        Returns:

        """

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path,
                                                           trust_remote_code=self.trust_remote_code,
                                                           padding_side=padding_side)
            # todo: some tokenizer may lost eos token, find some place to add to it.
            # if self.tokenizer.eos_token_id == None:
            #    self.tokenizer.eos_token_id = self.default_gen_cfg["eos_token_id"]
            # if self.tokenizer.pad_token_id == None:
            #    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        return self

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def read_model_config(self):
        """ read mode's model config from huggingface,
        and store in class, will auto call this function in load model. """

        if self.hf_model_config is None:
            self.hf_model_config = AutoConfig.from_pretrained(self.hf_model_path,
                                                              trust_remote_code=self.trust_remote_code)
            self.adapter = ModelAdapterFactory.create_adapter(self.hf_model_config)
            self.as_model_config = self.adapter.get_as_model_config()
            adapter = ModelAdapterFactory.create_adapter(self.hf_model_config)
            self.as_model_config = adapter.get_as_model_config()
            if self.user_set_data_type is None:
                self.data_type = adapter.get_model_data_type()
            print(f"model data type: {self.data_type}")

        # setup model data type by hf config.

        return self

    def create_reference_generation_config_builder(self, runtimeConfig: AsModelConfig = None):
        """
        create a new generate config for this request
        https://huggingface.co/docs/transformers/main_classes/text_generation

        if AsModelRuntimeConfig is set, will update max_length to engine max length in this api.

        Returns: self

        """
        try:
            generation_config = GenerationConfig.from_pretrained(self.hf_model_path,
                                                                 trust_remote_code=self.trust_remote_code)
            as_generate_config = ASGenerationConfigBuilder(generation_config)
        except Exception as e:
            print(f"create generation config from hf model failed with :{e}, use basic generation config")
            as_generate_config = ASGenerationConfigBuilder(eos_token_id=self.get_model_config()["eos_token_id"])

        if runtimeConfig:
            as_generate_config.update({"max_length": min(self.get_model_context_length(), runtimeConfig.engine_max_length)})

        # For JSON Mode
        if self.vocab is not None and self.vocab_type is not None:
            as_generate_config.update({"vocab": self.vocab})
            as_generate_config.update({"vocab_type": self.vocab_type})

        return as_generate_config

    def get_model_context_length(self):
        if self.hf_model_config is None:
            self.read_model_config()
        return self.as_model_config["max_position_embeddings"]

    def create_reference_runtime_config_builder(self, engine_model_name, device_type, device_ids, max_batch,
                                                max_context_length=-1) -> "AsModelRuntimeConfigBuilder":
        """
        it is a helper function to create a reference model runtime config by the information in model loader.
        you can also use AsModelRuntimeConfigBuilder to create runtime config.

        It will fill the model path by the information in serialize_to_path function.
        also will fill max context length by model support max length， you can use a smaller one.

        Args:
            engine_model_name: model name want to install into engine, normally it can same as model's filename,
                               but different name also ok
            device_type:  device want to engine to use.
            device_ids:   device 's device id.
            max_batch:   max batch size want engine to start.
            max_context_length: max context length of model, -1 means atomic detect by model's define,
                                for some small memory device, you may want to use a smaller length.


        Returns:  a runtime model config.

        """
        from .runtime_config import AsModelRuntimeConfigBuilder

        # if no max context provided, use model's context length
        if max_context_length == -1:
            max_context_length = self.get_model_context_length()

        if self.in_memory_serialize:
            runtime_cfg_builder = (AsModelRuntimeConfigBuilder()
                                   .model_name(engine_model_name)
                                   .model_file_path(self.temp_model_file.name, self.temp_weight_file.name)
                                   .compute_unit(device_type, device_ids)
                                   .max_length(max_context_length)
                                   .max_batch(max_batch)
                                   )
            return runtime_cfg_builder
        else:
            runtime_cfg_builder = (AsModelRuntimeConfigBuilder()
                                   .model_name(engine_model_name)
                                   .model_dir(self.model_output_dir, self.serialize_model_name)
                                   .compute_unit(device_type, device_ids)
                                   .max_length(max_context_length)
                                   .max_batch(max_batch)
                                   )
            return runtime_cfg_builder

    def free_model(self):
        "free all stored model file to save memory."
        self.torch_model = None
        self.torch_model_state_dict = None
        return self

    def free_memory_serialize_file(self):
        """
        free memory on serialize model to memory operation
        Returns:self

        """
        if self.temp_weight_file:
            self.temp_weight_file.close()
        if self.temp_model_file:
            self.temp_model_file.close()
        return self

    def export_model_diconfig(self, output_path):
        # generate a di config and save into local folder.
        from .config.diconfig import DIConfigBuilder
        from .runtime_config import AsModelRuntimeConfigBuilder
        from .engine import TargetDevice
        runtime_cfg_builder = (AsModelRuntimeConfigBuilder()
                               .model_name("model")
                               .model_file_path("", "")
                               .compute_unit(TargetDevice.CUDA, [0])
                               .max_length(2048)
                               .max_batch(64)
                               )
        runtime_config = runtime_cfg_builder.build()

        gen_config = self.create_reference_generation_config_builder(runtime_config)

        builder = DIConfigBuilder()
        (builder.model(self.as_model_config).runtime_config(runtime_cfg_builder.build())
         .generation_config(gen_config.build()).tokenizer().build_yaml_path(output_path))

        return self


    def serialize(self, engine, model_output_dir: str = "", enable_quant=False, weight_only_quant=False,
                  customized_quant_config=None, multinode_mode=True, lora_cfg=None):
        if self.in_memory_serialize:
            return self.serialize_to_memory(engine, enable_quant=enable_quant, weight_only_quant=weight_only_quant,
                                            multinode_mode=multinode_mode,
                                            customized_quant_config=customized_quant_config,
                                            lora_cfg=lora_cfg)
        else:
            if not model_output_dir:
                raise ValueError("model_output_dir is required for non-in-memory serialize")
            return self.serialize_to_path(engine, model_output_dir=model_output_dir, enable_quant=enable_quant,
                                          multinode_mode=multinode_mode,
                                          weight_only_quant=weight_only_quant,
                                          customized_quant_config=customized_quant_config,
                                          lora_cfg=lora_cfg)

    def serialize_to_memory(self, engine, enable_quant=False, weight_only_quant=False, multinode_mode=True, customized_quant_config=None, lora_cfg=None):
        """serialize the model, and save into memory file

        Returns: return two file path for in file tmp memory

        Args:
            engine:
            model_output_dir:
        """

        if not self.in_memory_serialize:
            raise Exception("Model loader not enable with memory serialize, please set HuggingFaceModel("
                            "in_memory_serialize=True)")

        # TODO: use actual memory, some tmp file is not backed memory.
        import tempfile
        import time
        begin = time.time()
        safe_model_name = self.pretain_model_name.replace("/", "_")

        self.temp_model_file = tempfile.NamedTemporaryFile()
        self.temp_weight_file = tempfile.NamedTemporaryFile()

        as_quant_config = None
        do_quant_convert = enable_quant

        if enable_quant:
            if customized_quant_config:
                as_quant_config = self.get_as_quant_config_and_weight_process(weight_only_quant=weight_only_quant, customized_quant_config=customized_quant_config)
            else:
                as_quant_config = self.get_as_quant_config_and_weight_process(weight_only_quant=weight_only_quant)

        engine.serialize_model_from_torch(model_name=safe_model_name,
                                          model_type=self.as_model_config["model_type"],
                                          torch_model=self.torch_model_state_dict,
                                          model_config=self.as_model_config,
                                          do_dynamic_quantize_convert=do_quant_convert,
                                          data_type=self.data_type,
                                          multigpu_mode=multinode_mode,
                                          rotary_base=self.as_model_config["rotary_emb_base"],
                                          quant_config=as_quant_config,
                                          as_model_path=self.temp_model_file.name,
                                          as_weight_path=self.temp_weight_file.name,
                                          lora_cfg=lora_cfg)
        # free all gpu memory during serialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.serialize_model_name = safe_model_name
        sec = time.time() - begin
        print(f"Model {self.pretain_model_name} serialize finished, consume {sec} seconds...")

        return self

    def serialize_to_path(self, engine, model_output_dir: str = "", enable_quant=False, weight_only_quant=False,
                          skip_if_exists=False, multinode_mode=True, customized_quant_config=None, lora_cfg=None):
        """
        start serialize loaded model and write to dir, and free the torch model to save memory.
        Args:
            engine: engine instance
            model_output_dir: the serialize model dir.
            enable_quant: enable quantization config, will read model's quant related config
            weight_only_quant: enable weight only quant or activation & weight quant,
                    weight only quant means use BF16 or FP16 compute, the weight will dequantization to 16bit on-the-fly
                    activation & weight quant means use lower bit to compute, you will get more computer power in this mode.

            skip_if_exists:  skip serialize if model serialize file already exists, default is False, the serialize will
                     happen every call to this function.
            with_di_config: create a diformat config for di model loader quick load.

        Returns: self
        """
        import time
        begin = time.time()

        if len(model_output_dir) == 0:
            raise ValueError(f"model_output_dir should not be empty string: {model_output_dir}")

        safe_model_name = self.pretain_model_name.replace("/", "_")
        model_suffix = "..asgraph"
        weights_suffix = ".asparam"

        model_path = os.path.join(model_output_dir, safe_model_name + model_suffix)
        weights_path = os.path.join(model_output_dir, safe_model_name + weights_suffix)
        really_skip_convert = False
        if skip_if_exists:
            if os.path.exists(model_path) and os.path.exists(weights_path):
                if (os.path.getsize(model_path) > 0) and (os.path.getsize(weights_path) > 0):
                    really_skip_convert = True
                    print("serialize_to_path: skip_if_exists = True, Skip Model serialize because model file seems "
                          "exists.")

        as_quant_config = None
        do_quant_convert = enable_quant

        if enable_quant and not really_skip_convert:
            if customized_quant_config:
                as_quant_config = self.get_as_quant_config_and_weight_process(weight_only_quant=weight_only_quant,
                                                                              customized_quant_config=customized_quant_config)
            else:
                as_quant_config = self.get_as_quant_config_and_weight_process(weight_only_quant=weight_only_quant)

        if not really_skip_convert:
            engine.serialize_model_from_torch(model_name=safe_model_name,
                                              model_type=self.as_model_config["model_type"],
                                              torch_model=self.torch_model_state_dict,
                                              model_config=self.as_model_config,
                                              do_dynamic_quantize_convert=do_quant_convert,
                                              multigpu_mode=multinode_mode,
                                              data_type=self.data_type,
                                              rotary_base=self.as_model_config["rotary_emb_base"],
                                              quant_config=as_quant_config,
                                              save_dir=model_output_dir,
                                              lora_cfg=lora_cfg)

        # free all gpu memory during serialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_output_dir = model_output_dir
        self.serialize_model_name = safe_model_name
        sec = time.time() - begin
        print(f"Model {self.pretain_model_name} serialize finished, consume {sec} seconds...")

        return self

    def get_as_quant_config_and_weight_process(self, weight_only_quant: bool = False, customized_quant_config = None):
        """
        GPTQ currently don't have config to specify the sign flag of weight, so for gptq model,
        allspark will process the weight to known sign, and call to generate the correct as quant config.

        Returns: Engine's quant config with correct subchannel settings.

        """
        # setup self.quant_settings
        self.init_quantization(customized_quant_config=customized_quant_config)
        activation_data_type = self.data_type

        if customized_quant_config:
            orig_val = weight_only_quant
            # default use weight only quant mode.
            if customized_quant_config.get("compute_method", "weight_only") == "weight_only":
                weight_only_quant = True
            else:
                weight_only_quant = False
            if orig_val != weight_only_quant:
                print(f"Notice: customized quant config was given, the user weight_only_quant will be override: orig: {orig_val}, new: {weight_only_quant}")

        if weight_only_quant:
            activation_data_type = self.data_type
        elif self.quant_setting.get_name() == "instant_quant":
            activation_data_type = self.quant_setting.weight_format
        else:
            activation_data_type = "int8"
            # for now, true quant only support int8, not include uint8
            # for int4, activate quant only can use int8,
            # TODO: how to add fp8 is a issue.

        if self.quant_setting.get_name() == "gptq":
            is_subchannel = self.quant_setting.get_group_num() != -1
            if is_subchannel:
                adapter = GPTQ2IQWeightAdapter()
                self.torch_model_state_dict = adapter.dequant_gptq_weight(self.torch_model_state_dict,
                                                                          self.quant_setting.get_weight_bits(),
                                                                          model_dtype=self.data_type)
                weight_type = "int8"
                if weight_only_quant:
                    quant_extra_config = {"SubChannel": is_subchannel,
                                          "GroupSize": self.quant_setting.get_group_num()}
                else:
                    # because a8w8 only support per-channel, setting group setting in here.
                    quant_extra_config = {"SubChannel": False,
                                          "GroupSize": -1
                                          }

                if self.quant_setting.get_weight_bits() == 4:
                    weight_type = "uint4"
                    quant_extra_config = {"SubChannel": False,
                                          "GroupSize": self.quant_setting.get_group_num()
                                          }
            else:
                # model weight provided by yaoyang, each tensor is int8 and store in float16 format. Only support A8W8 
                assert self.quant_setting.get_weight_bits() == 8, "Perchannel weight only support A8W8"
                assert weight_only_quant == False, "Perchannel weight only support A8W8, not support A8W16"
                weight_type = "int8"
                quant_extra_config = {
                    "SubChannel": False,
                    "GroupSize": -1,
                    "AdaptedQuantMethod": "GPTQ_NO_PACK"
                }

            as_quant_config = QuantizeConfig(activation_type=activation_data_type, weight_type=weight_type,
                                             extra_option=quant_extra_config)
            print(f"QuantConfig: {as_quant_config}")
            return as_quant_config

        elif self.quant_setting.get_name() == "instant_quant":
            as_quant_config = self.quant_setting.to_as_quant_config(activation_data_type, self.quant_setting.weight_format)
            print(f"QuantConfig: {as_quant_config}")

            return as_quant_config
        else:
            raise ValueError(f"unknown quant method: {self.quant_setting.get_name()}")

    def get_model_config(self):
        """get model config for build_model_from_config_struct interface."""

        return self.as_model_config
