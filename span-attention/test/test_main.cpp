/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    test_main.cpp
 */
#include "test_common.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
