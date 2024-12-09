'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    model_loader.py
'''
import os
import torch
import glob
from modelscope import snapshot_download
from transformers import Qwen2VLForConditionalGeneration, AutoConfig, AutoTokenizer
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from tqdm import tqdm
from safetensors.torch import safe_open
from dashinfer import allspark
from dashinfer.allspark.model_loader import HuggingFaceModel, ModelSerializerException
from dashinfer.allspark.model_config import QWen2ConfigAdapter
from .trt.onnx_to_plan import ONNX_TRT


class HuggingFaceVLModel(HuggingFaceModel):
    def __init__(
        self,
        pretrain_model_name_or_path,
        pretrain_model_name="model",
        in_memory_serialize=False,
        user_set_data_type="bfloat16",
        trust_remote_code=True,
        vision_engine="hie",
        fp8=False
    ):
        super().__init__(
            pretrain_model_name_or_path,
            pretrain_model_name,
            in_memory_serialize,
            user_set_data_type,
            trust_remote_code,
        )

        self.vision_engine = vision_engine
        self.fp8 = fp8

    def load_model(
        self, override_data_type=None, direct_load=False, load_format="auto", **kwargs
    ):
        if not direct_load:
            # the open-source model can be loaded by huggingface
            try:
                if not os.path.isdir(self.hf_model_path):
                    self.hf_model_path = snapshot_download(self.hf_model_path)
                self.torch_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.hf_model_path,
                    trust_remote_code=self.trust_remote_code,
                    **kwargs,
                ).eval()
                self.vit_config = Qwen2VLVisionConfig.from_pretrained(
                    self.hf_model_path,
                    trust_remote_code=True,
                    revision=None,
                    code_revision=None,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_model_path,
                    trust_remote_code=self.trust_remote_code,
                    **kwargs,
                )
                self.torch_model = self.torch_model.cpu()

                if self.data_type == "float32":
                    self.torch_model.float()
                elif self.data_type == "float16":
                    self.torch_model.half()
                elif self.data_type == "bfloat16":
                    self.torch_model.bfloat16()
                else:
                    self.torch_model = None
                    raise ValueError("unsupported data type: {}".format(self.data_type))
            except Exception as e:
                print(
                    f"exception when load model: {self.hf_model_path} , exception: {e}"
                )
                raise ModelSerializerException("error load from huggingface", 1, self)

            # allspark only needs the torch's model's state dict, no needs to keep torch model
            self.torch_model_state_dict = self.torch_model.state_dict()
            self.torch_model = None
        else:
            NotImplementedError("direct_load from vl is not implemented!")

        self.read_model_config()

        return self

    def read_model_config(self):
        """read mode's model config from huggingface,
        and store in class, will auto call this function in load model."""

        if self.hf_model_config is None:
            self.hf_model_config = AutoConfig.from_pretrained(
                self.hf_model_path, trust_remote_code=self.trust_remote_code
            )
            self.adapter = QWen2ConfigAdapter(self.hf_model_config)
            self.as_model_config = self.adapter.model_config
            if self.user_set_data_type is None:
                self.data_type = self.adapter.get_model_data_type()
            print(f"model data type: {self.data_type}")
        return self

    def serialize(
        self, model_output_dir: str = ""
    ):
        if model_output_dir == "":
            raise NotImplementedError("Not support serialize without model_output_dir")
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        engine = allspark.Engine()

        if self.vision_engine == "tensorrt":
            onnxFile = os.path.join(model_output_dir, self.pretain_model_name + ".onnx")
            self.vision_model_path = os.path.join(
                model_output_dir, self.pretain_model_name + ".plan"
            )
            onnx_trt_obj = ONNX_TRT(self.hf_model_path)
            onnx_trt_obj.export_onnx(onnxFile)
            onnx_trt_obj.generate_trt_engine(onnxFile, self.vision_model_path)
        else:
            raise ValueError(f"unsupported engine {self.vision_engine}")

        # Convert Allspark LLM
        enable_quant = self.fp8
        weight_only_quant=False
        quant_config = None
        if self.fp8:
            quant_config = {"quant_method": "instant_quant", "weight_format": "fp8_e4m3", "compute_method": "activate_quant"}
        return super().serialize(
            engine, model_output_dir, enable_quant, customized_quant_config=quant_config, weight_only_quant=weight_only_quant
        )

    def onnx_export(self, model, onnx_fname, image, grid_thw, batch):
        onnx_dir = os.path.dirname(onnx_fname)
        if not os.path.exists(onnx_dir):
            os.makedirs(onnx_dir)

        # wrap the model
        class WarpModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, image, grid_thw, batch):
                return self.model(image, grid_thw, batch)

        wrap_model = WarpModel(model)
        wrap_model.eval()
        with torch.no_grad():
            torch.onnx.export(
                wrap_model,
                (image, grid_thw, batch),
                onnx_fname,
                input_names=["image", "grid_thw", "batch"],
                output_names=["output"],
                dynamic_axes={
                    "image": {0: "height", 1: "width"},
                    "grid_thw": {0: "batch", 1: "thw"},
                    "batch": {0: "batch"},
                    "output": {0: "img_seqlens", 1: "num_classes"},
                },
                do_constant_folding=True,
                opset_version=11,
            )

    def get_weights_iterator(self, model):
        hf_folder, hf_weights_files, use_safetensors = self.prepare_weights(model)
        if use_safetensors:
            weights_iterator = self.safetensors_weights_iterator(hf_weights_files)
        else:
            raise ValueError
        return weights_iterator

    def safetensors_weights_iterator(self, hf_weights_files):
        """Iterate over the weights in the model safetensor files."""
        enable_tqdm = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        # hf_weights_files = sorted(hf_weights_files)
        for st_file in tqdm(
            hf_weights_files,
            desc="Loading safetensors checkpoint shards",
            disable=not enable_tqdm,
            # bar_format=_BAR_FORMAT,
        ):
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param

    def prepare_weights(self, model):
        allow_patterns = ["*.safetensors", "*.bin"]
        hf_weights_files = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(model, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break
        return model, hf_weights_files, use_safetensors
