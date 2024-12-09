'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_di_model_config.py
'''
import gc
import time
import unittest

import modelscope

class DIModelConfigTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_yaml_dict(self):
        import sys
        from ruamel.yaml import YAML

        yaml_str = """\
        first_name: Art
        occupation: Architect  # This is an occupation comment
        about: Art Vandelay is a fictional character that George invents...
        """

        yaml = YAML()
        data = yaml.load(yaml_str)
        data.insert(0, 'first_name', 'some name', comment="name comments")
        data.insert(1, 'last name', 'Vandelay', comment="new key")


        yaml.dump(data, sys.stdout)


        import sys
        import ruamel.yaml

        yaml = ruamel.yaml.YAML()  # defaults to round-trip

        inp = """\
        abc:
          - a     # comment 1
        xyz:
          a: 1    # comment 2
          b: 2
          c: 3
          d: 4
          e: 5
          f: 6 # comment 3
        """

        data = yaml.load(inp)
        data['abc'].append('b')
        data['abc'].yaml_add_eol_comment('comment 4', 1)  # takes column of comment 1
        data['xyz'].yaml_add_eol_comment('comment 5', 'c')  # takes column of comment 2
        data['xyz'].yaml_add_eol_comment('comment 6', 'e')  # takes column of comment 3
        data['xyz'].yaml_add_eol_comment('comment 7\n\n# that\'s all folks', 'd', column=20)

        yaml.dump(data, sys.stdout)


if __name__ == '__main__':
    unittest.main()
