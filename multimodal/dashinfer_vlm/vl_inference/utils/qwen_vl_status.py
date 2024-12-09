'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    qwen_vl_status.py
'''
class VLStatusCode:
    VL_SUCCESS = 0
    VL_REQUEST_ERROR = 1
    VL_FILE_ERROR = 2
    VL_IMAGE_FORMAT_ERROR = 3
    VL_VIT_NUMS_MISMATCH_ERROR = 4
    VL_INPUT_FORMAT_ERROR = 5
    VL_VISION_DECODE_ERROR = 6
    VL_INPUT_TOKENS_ERROR = 7
    VL_OTHER_ERROR = 100


class VLResult:
    def __init__(
        self,
        status: VLStatusCode = VLStatusCode.VL_SUCCESS,
        error: Exception = None,
        error_details: str = "",
    ):
        self._status = status
        self._error = error
        self.error_details = error_details

    @property
    def status(self):
        return self._status

    @property
    def error(self):
        return self._error

    @property
    def error_details(self):
        return self._error_details

    @status.setter
    def status(self, value):
        self._status = value

    @error.setter
    def error(self, value):
        self._error = value

    @error_details.setter
    def error_details(self, value):
        self._error_details = value


class Interval:

    def __init__(self, type: str = "", cost: int = 0):
        self._type = type
        self._cost = cost

    @property
    def type(self):
        return self._type

    @property
    def cost(self):
        return self._cost

    @type.setter
    def type(self, value):
        self._type = value

    @cost.setter
    def cost(self, value):
        self._cost = value
