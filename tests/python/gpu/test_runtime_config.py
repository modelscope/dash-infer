'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_runtime_config.py
'''
# 引入必要的类和方法
from dashinfer.allspark._allspark import AsModelConfig

from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder
import unittest

# 定义测试类
class TestAsModelRuntimeConfigBuilder(unittest.TestCase):

    # 定义测试'from_dict'方法的正常行为
    def test_from_dict_normal_behavior(self):
        # 构造预期结果
        expected_model_config = AsModelConfig()
        expected_model_config.model_name = "test_model"
        expected_model_config.compute_unit = "CUDA:0,1"
        expected_model_config.engine_max_length = 100
        expected_model_config.engine_max_batch = 32
        expected_model_config.num_threads = 2

        # 构造输入字典
        input_dict = {
            'model_name': 'test_model',
            'compute_unit': {
                'device_type': 'cuda',
                'device_ids': [0, 1],
                'compute_thread_in_device': 2
            },
            'engine_max_length': 100,
            'engine_max_batch': 32
        }

        # 创建Builder实例并调用from_dict方法
        builder = AsModelRuntimeConfigBuilder()
        builder.from_dict(input_dict)

        # 获取实际结果并进行断言
        actual_model_config = builder.build()  # Assuming there is a to_config method to get the final config
        print("expect model config: ", expected_model_config)
        print("model config : ", actual_model_config)
        self.assertEqual(str(expected_model_config), str(actual_model_config))

# 运行测试
if __name__ == '__main__':
    unittest.main()
