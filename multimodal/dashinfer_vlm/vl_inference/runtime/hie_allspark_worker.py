'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    hie_allspark_worker.py
'''
from ..utils.hie_allspark.model_hie_allspark import (
    AllSparkM6Model,
)
from ..utils.hie_allspark.model_hie_allspark import (
    AllSparkRequest,
)
from dashinfer import allspark
import torch


class HieAllsparkWorker:
    def __init__(self, as_model_config: allspark.AsModelConfig):
        self.model = AllSparkM6Model(as_model_config)

    def eval(self, request: AllSparkRequest):
        # qwenvl2 to get position list
        position_list = self.get_llm_positions(
            request.input_lists, request.vit_positions, request.vit_target_token
        )
        request.gen_cfg = self.get_gen_cfg(request, position_list)
        return self.model.forward(request)

    """
    compute llm positions with mrope
    """

    def get_llm_positions(
        self,
        total_input_ids,
        vit_grid_thw_list,
        image_modality_token_id,
    ):
        spatial_merge_size = 2
        assert len(total_input_ids) == len(vit_grid_thw_list)
        total_llm_positions = []

        for input_tokens, grid_thw in zip(total_input_ids, vit_grid_thw_list):
            if len(grid_thw) > 0 and grid_thw[0] is None:
                return []
            llm_pos_ids_list: list = []
            st = 0
            for t, h, w in grid_thw:
                ed = input_tokens.index(image_modality_token_id, st)
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max().item() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                llm_pos_ids_list.append(torch.arange(text_len).repeat(3, 1) + st_idx)

                _llm_tpos_ids = (
                    torch.arange(llm_grid_t)
                    .unsqueeze(1)
                    .repeat(1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                _llm_hpos_ids = (
                    torch.arange(llm_grid_h)
                    .unsqueeze(1)
                    .repeat(1, llm_grid_w)
                    .flatten()
                    .repeat(llm_grid_t)
                )
                _llm_wpos_ids = (
                    torch.arange(llm_grid_w)
                    .unsqueeze(0)
                    .repeat(llm_grid_h, 1)
                    .flatten()
                    .repeat(llm_grid_t)
                )
                _llm_pos_ids = torch.stack(
                    [_llm_tpos_ids, _llm_hpos_ids, _llm_wpos_ids]
                )
                llm_pos_ids_list.append(_llm_pos_ids + text_len + st_idx)
                st = ed + _llm_pos_ids.shape[-1]

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max().item() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).repeat(3, 1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(
                3, len(input_tokens)
            )
            total_llm_positions.append(llm_positions)

        total_llm_positions = torch.stack(total_llm_positions, dim=1)
        return total_llm_positions

    def get_gen_cfg(self, request: AllSparkRequest, position_list: list):
        dl_list = []
        pos_dl_list = []
        # modify gen_cfg max_length
        # since truncate is done in 一体化, we need to modify max_length
        # logging.error(f"request.gen_cfg max_length:{request.gen_cfg['max_length']}, old_context_len:{request.old_context_len}, new_context_len:{request.new_context_len}")
        request.gen_cfg["max_length"] = (
            request.gen_cfg["max_length"]
            - request.old_context_len
            + request.new_context_len
        )
        if request.gen_cfg["max_length"] > request.max_total_tokens:
            request.gen_cfg["max_length"] = request.max_total_tokens
        if len(request.vit_embs) == 0:
            return request.gen_cfg
        for vit in request.vit_embs:
            dl_list.append(torch.utils.dlpack.to_dlpack(vit.to(torch.float32)))
        if len(position_list) > 0:
            pos_dl_list.append(
                torch.utils.dlpack.to_dlpack(position_list.to(torch.int32))
            )
        as_extra_embedding_info_0 = allspark.MultiMediaInfo()
        as_extra_embedding_info_0.set_multimedia_type(0)
        as_extra_embedding_info_0.add_multimedia_content(
            str(request.vit_target_token), dl_list
        )
        key_list = []
        if request.vit_keys is not None:
            for key in request.vit_keys:
                # logging.info(f"key shape: {key.shape}, key:{key}")
                key_list.append(torch.utils.dlpack.to_dlpack(key.to(torch.int32)))
            as_extra_embedding_info_0.add_multimedia_content("hash_input", key_list)
        if len(pos_dl_list) > 0:
            as_extra_embedding_info_0.add_multimedia_content("positions", pos_dl_list)
        request.gen_cfg["mm_info"] = as_extra_embedding_info_0
        request.gen_cfg["extra_embedding"] = dl_list
        request.gen_cfg["extra_embedding_pos"] = pos_dl_list
        request.gen_cfg["extra_embedding_key"] = key_list
        return request.gen_cfg

    def terminate(self):
        self.model.terminate()
