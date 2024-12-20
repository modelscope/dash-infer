'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    detect-lora-limit.py
'''
import os, pdb
import multiprocessing
import sys
import torch

from dashinfer import allspark
from dashinfer.allspark import AsStatus, AsModelConfig
from dashinfer.allspark.engine import TargetDevice
import argparse


global arg


def check_proc(lora_num, runtime_cfg_dict):
    global args
    model_name = runtime_cfg_dict['model_name']

    print('trying lora_num=', lora_num)
    runtime_cfg = AsModelConfig(model_name, runtime_cfg_dict['model_path'], runtime_cfg_dict['weights_path'], runtime_cfg_dict['compute_unit'], args.max_length, args.max_batch)
    runtime_cfg.lora_max_rank = args.lora_max_rank
    runtime_cfg.lora_max_num = lora_num

    ret = AsStatus.ALLSPARK_SUCCESS
    try:
        engine = allspark.Engine()
        engine.install_model(runtime_cfg)
        ret = engine.start_model(model_name)
        engine.stop_model(model_name)
        engine.release_model(model_name)
    except BaseException as e:
        print(str(e))
        return False

    if ret != AsStatus.ALLSPARK_SUCCESS:
        print('start_model Error!', ret)
        return False
    print('lora_num=', lora_num, 'passed!')
    return True

def child_process(read_fd, write_fd, lora_num, runtime_cfg_dict):
    os.close(read_fd)
    ret = check_proc(lora_num, runtime_cfg_dict)
    os.write(write_fd, str(ret).encode())
    os.close(write_fd)
    return ret
 

def multi_process_check(lora_num, runtime_cfg_dict):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid < 0:
        print('fork failed, exit!')
        os._exit(1)
    if pid == 0:  # child process
        child_process(read_fd, write_fd, lora_num, runtime_cfg_dict)
        os._exit(0)

    # father process
    os.close(write_fd)
    child_ret = None
    try:
        data = os.read(read_fd, 100)
        if data:
            child_ret = data.decode().strip()
        os.waitpid(pid, 0)
    except:
        child_ret = None
        print('child crash!')
        os.waitpid(pid, 0)
        os.close(read_fd)
        return False
    if child_ret is None or child_ret != 'True':
        os.close(read_fd)
        return False
    os.close(read_fd)
    return True
     

def binary_search(min_num, max_num, runtime_cfg_dict):
    left = min_num
    right = max_num

    while left <= right:
        mid = (left + right) // 2
        passed = multi_process_check(mid, runtime_cfg_dict)
        if passed:
            left = mid + 1
        else:
            right = mid - 1
    return mid if passed else (mid -1)


def save_as_model(runtime_cfg_dict):
    global args
    pid = os.fork()
    if pid < 0:
        print('fork failed, exit!')
        os._exit(1)
    if pid > 0:  # father process
        os.waitpid(pid, 0)
        print('=============================================save OK')
        return

    # child process

    # specified by user cmdline args:
    init_quant = args.quantization
    device_list = list(range(args.parallel))

    base_model_dir = runtime_cfg_dict['base_model_dir']
    output_model_dir = runtime_cfg_dict['output_model_dir']
    model_name = runtime_cfg_dict['model_name']
    in_memory = runtime_cfg_dict['in_memory']
    weight_only_quant = runtime_cfg_dict['weight_only_quant']

    model_loader = allspark.HuggingFaceModel(base_model_dir, model_name, user_set_data_type="float16", in_memory_serialize=in_memory, trust_remote_code=True)
    engine = allspark.Engine()
    if in_memory:
        (model_loader.load_model()
         .read_model_config()
         .serialize_to_memory(engine, enable_quant=init_quant, weight_only_quant=weight_only_quant, lora_cfg={},
                            customized_quant_config= {"quant_method": "instant_quant", "weight_format": "int8"},)
         .free_model())
    else:
        (model_loader.load_model()
         .read_model_config()
         .serialize_to_path(engine, output_model_dir, enable_quant=init_quant, weight_only_quant=weight_only_quant,
                            skip_if_exists=True, lora_cfg={},
                            customized_quant_config= {"quant_method": "instant_quant", "weight_format": "int8"},
                            )
         .free_model())
    engine = None
    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(model_name, TargetDevice.CUDA,
                                                                               device_list, max_batch=args.max_batch)
    runtime_cfg_builder.max_length(args.max_length).lora_max_rank(args.lora_max_rank)
    runtime_cfg = runtime_cfg_builder.build()
    runtime_cfg_dict['model_name'] = runtime_cfg.model_name
    runtime_cfg_dict['model_path'] = runtime_cfg.model_path
    runtime_cfg_dict['weights_path'] = runtime_cfg.weights_path
    runtime_cfg_dict['engine_max_length'] = runtime_cfg.engine_max_length
    runtime_cfg_dict['engine_max_batch'] = runtime_cfg.engine_max_batch
    runtime_cfg_dict['compute_unit'] = runtime_cfg.compute_unit
    os._exit(0)


def main():
    global args
    os.environ['ALLSPARK_DISABLE_WARMUP'] = '0'
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--base_model_dir", type=str, required=True, help="dir of base model")
    parser.add_argument("-r", "--lora_max_rank", type=int, required=True, help="lora max rank")
    parser.add_argument("-q", "--quantization", action="store_true", help="true if toggled, which means quantizing base model to A16W8")
    parser.add_argument("-p", "--parallel", type=int, required=True, help="num of GPU cards")
    parser.add_argument("-b", "--max_batch", type=int, required=True, help="max batchsize")
    parser.add_argument("-l", "--max_length", type=int, required=True, help="max length")
    args = parser.parse_args()
    print(args)

    # you can also change these parameters manually:
    in_memory = True
    weight_only_quant = True

    shared_manager = multiprocessing.Manager()
    runtime_cfg_dict = shared_manager.dict()
    runtime_cfg_dict['base_model_dir'] = args.base_model_dir   # 基模目录
    runtime_cfg_dict['output_model_dir'] = "model_output.lora-detect"   # in_memory==False时，.asparam模型的目录
    runtime_cfg_dict['model_name'] = "my_test_model" #.asparam模型名字
    runtime_cfg_dict['in_memory'] = in_memory
    runtime_cfg_dict['weight_only_quant'] = weight_only_quant

    save_as_model(runtime_cfg_dict)
    print(runtime_cfg_dict)


    # 二分搜索的范围
    lower_lora_num = 1
    upper_lora_num = 100
    result = binary_search(lower_lora_num, upper_lora_num, runtime_cfg_dict)
    print('Final detection result: lora_max_num=', result)
    
    if runtime_cfg_dict['in_memory']:
        os.unlink(runtime_cfg_dict['weights_path'])
        os.unlink(runtime_cfg_dict['model_path'])

main()
