#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    arbiter_fileclient.py
#
import os
import re
import numpy as np
import torch
import pdb
# from bfloat16 import bfloat16


class arbiter:
    __layer_idx = -1
    __seq_len = 0
    __npy_base_dir = '/root/workspace/ALLSPARK_DUMP'
    __mode = 0  # 0: strict-mode    1: tolerance-mode

    @staticmethod
    def set_mode(mode):
        arbiter.__mode = mode

    @staticmethod
    def update_layer_idx(layer_idx):
        arbiter.__layer_idx = layer_idx

    @staticmethod
    def update_seq_len():  # seq_len==0 表示位于Context阶段
        arbiter.__seq_len = arbiter.__seq_len + 1

    @staticmethod
    def reset_seq_len():
        arbiter.__seq_len = 0

    @staticmethod
    def incr_seq_len(delta_len):
        if not arbiter.__seq_len:  # in context-phase, incr result should + 1
            arbiter.__seq_len = 1
        arbiter.__seq_len += delta_len

    @staticmethod
    def do_arbitrate_prev_layer(t_name,
                                torch_t,
                                orig_tensors=None,
                                transpose_info=None):
        if '__LAYER__' in t_name and arbiter.__layer_idx == -1:
            raise Exception('update_layer_idx first!!!')
        if arbiter.__layer_idx == 0:
            raise Exception('layer 0 has no prev layer!!!')

        t = torch_t
        prev_layer = arbiter.__layer_idx - 1
        as_t_name = re.sub(r'__LAYER__', f'decoder.layer.{prev_layer}', t_name)
        use_bf16 = False
        if isinstance(t, torch.Tensor):
            if t.dtype == torch.bfloat16:
                #raise Exception('pytorch is using bf16, not supported in arbitration mode!!!')
                t = t.to('cpu').float().numpy().astype(bfloat16)
                use_bf16 = True
            else:
                t = t.to('cpu').numpy()

        sub_dir = f'seq_len_{arbiter.__seq_len}' if arbiter.__seq_len > 0 else 'context_phase'
        npy_dir = os.path.join(arbiter.__npy_base_dir, 'to_be_verified',
                               sub_dir)
        npy_file = os.path.join(npy_dir, as_t_name + '.npy')
        as_npy = np.load(npy_file)
        if use_bf16:
            as_npy.dtype = bfloat16
            as_npy = np.copy(as_npy)

        match = np.isclose(t.astype(np.float32),
                           as_npy.astype(np.float32),
                           atol=0.005).all()
        mse = np.mean(np.square(t - as_npy))
        print(
            f't_name={t_name} seq_len={arbiter.__seq_len} layer_idx={arbiter.__layer_idx} mse={mse}'
        )
        if not match or mse > 0.001:
            ref_npy_dir = os.path.join(arbiter.__npy_base_dir, 'reference',
                                       sub_dir)
            ref_tensor_path = os.path.join(ref_npy_dir, f'{as_t_name}.npy')
            os.makedirs(ref_npy_dir, exist_ok=True)
            np.save(ref_tensor_path, np.ascontiguousarray(t))
            print(
                as_t_name,
                f'prev NOT match!!! Reference tensor saved to {ref_tensor_path}'
            )
            # 如果只要有一个不匹配就退出的话，使用exit
            # exit()

        if arbiter.__mode == 1:
            if orig_tensors:
                if len(orig_tensors) == 1:
                    orig_t = orig_tensors[0]
                    t = torch.from_numpy(as_npy.astype(np.float32))
                    if transpose_info:
                        t = t.transpose(*transpose_info)
                    assert t.shape == orig_t.shape
                    orig_t.copy_(t.reshape(orig_t.shape).to(orig_t.dtype))
                else:
                    assert (len(orig_tensors) == 3)
                    as_npy_qkv = np.split(as_npy, 3, axis=-1)
                    for i in range(3):
                        orig_t = orig_tensors[i]
                        t = torch.from_numpy(as_npy.astype(np.float32))
                        if transpose_info:
                            t = t.transpose(*transpose_info)
                        assert t.shape == orig_t.shape
                        orig_t.copy_(t.reshape(orig_t.shape).to(orig_t.dtype))
            else:
                torch_t.copy_(
                    torch.from_numpy(as_npy.astype(np.float32)).to(
                        torch_t.dtype))

    @staticmethod
    def do_arbitrate(t_name, torch_t, orig_tensors=None, transpose_info=None):
        if '__LAYER__' in t_name and arbiter.__layer_idx == -1:
            raise Exception('update_layer_idx first!!!')

        t = torch_t
        as_t_name = re.sub(r'__LAYER__',
                           f'decoder.layer.{arbiter.__layer_idx}', t_name)
        use_bf16 = False
        if isinstance(t, torch.Tensor):
            if t.dtype == torch.bfloat16:
                #raise Exception('pytorch is using bf16, not supported in arbitration mode!!!')
                t = t.to('cpu').float().numpy().astype(bfloat16)
                use_bf16 = True
            else:
                t = t.to('cpu').numpy()

        sub_dir = f'seq_len_{arbiter.__seq_len}' if arbiter.__seq_len else 'context_phase'
        ref_npy_dir = os.path.join(arbiter.__npy_base_dir, 'reference',
                                   sub_dir)
        ref_tensor_path = os.path.join(ref_npy_dir, f'{as_t_name}.npy')
        os.makedirs(ref_npy_dir, exist_ok=True)
        np.save(ref_tensor_path, np.ascontiguousarray(t))

        npy_dir = os.path.join(arbiter.__npy_base_dir, 'to_be_verified',
                               sub_dir)
        npy_file = os.path.join(npy_dir, as_t_name + '.npy')
        if not os.path.exists(npy_file):
            print(as_t_name, f'{npy_file} NOT EXISTS!!!')
            return

        as_npy = np.load(npy_file)
        if use_bf16:
            as_npy.dtype = bfloat16
            as_npy = np.copy(as_npy)
        match = np.isclose(t.astype(np.float32),
                           as_npy.astype(np.float32),
                           atol=0.005).all()
        mse = np.mean(np.square(t - as_npy))
        print(
            f'as_t_name={as_t_name} seq_len={arbiter.__seq_len} layer_idx={arbiter.__layer_idx} mse={mse}'
        )
        if not match or mse > 0.001:
            print(as_t_name,
                  f'NOT match!!! Reference tensor saved to {ref_tensor_path}')
            # 如果只要有一个不匹配就退出的话，使用exit
            # exit()

        if arbiter.__mode == 1:
            if orig_tensors:
                if len(orig_tensors) == 1:
                    orig_t = orig_tensors[0]
                    t = torch.from_numpy(as_npy.astype(np.float32))
                    if transpose_info:
                        t = t.transpose(*transpose_info)
                    assert t.shape == orig_t.shape
                    orig_t.copy_(t.reshape(orig_t.shape).to(orig_t.dtype))
                else:
                    assert (len(orig_tensors) == 3)
                    as_npy_qkv = np.split(as_npy, 3, axis=-1)
                    for i in range(3):
                        orig_t = orig_tensors[i]
                        t = torch.from_numpy(as_npy.astype(np.float32))
                        if transpose_info:
                            t = t.transpose(*transpose_info)
                        assert t.shape == orig_t.shape
                        orig_t.copy_(t.reshape(orig_t.shape).to(orig_t.dtype))
            else:
                torch_t.copy_(
                    torch.from_numpy(as_npy.astype(np.float32)).to(
                        torch_t.dtype))


'''
使用方法： 在transformers的modeling_xxx.py文件中：
```
from dashinfer.allspark.arbiter_fileclient import arbiter

# 模型每层循环里调用arbiter.update_layer_idx
class XXXModel():
    def forward(...):
        对每层循环:
            arbiter.update_layer_idx(layer_idx)
        return前调用 arbiter.update_seq_len(seq_len) # seq_len可以从kvcache获取，不同的模型名字不一样

# 每个sub block里调用arbiter.do_arbitrate
class XXXAttention():
    def forward(...):
        # 在计算出想要校验的tensor之后, 插入do_arbitrate， 例如：
        arbiter.do_arbitrate('__LAYER__.attention.out', attn_output)
    
```
'''
