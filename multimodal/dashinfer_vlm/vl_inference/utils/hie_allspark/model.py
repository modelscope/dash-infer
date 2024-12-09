'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    model.py
'''
import abc


class Model(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "forward")
            and callable(subclass.forward)
            or NotImplemented
        )

    @abc.abstractmethod
    def forward():
        raise NotImplementedError
