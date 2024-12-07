/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    test_common.cpp
 */

#include <test_common.h>
using namespace allspark;
allspark::AsStatus LoadBinFromFile(const std::string& src_file,
                                   void* cpu_tensor, int size) {
  FILE* fp = fopen(src_file.c_str(), "rb");
  if (!fp) {
    printf("[ERROR] file %s cannot be opened.\n", src_file.c_str());
    return allspark::AsStatus::ALLSPARK_IO_ERROR;
  }
  fseek(fp, 0, SEEK_END);
  int buf_size = ftell(fp);
  if (buf_size != size) {
    printf("[ERROR] buf size incompatible: %d vs %d, invalid data:%s\n",
           buf_size, size, src_file.c_str());
    fclose(fp);
    return allspark::AsStatus::ALLSPARK_IO_ERROR;
  }
  rewind(fp);
  fread(cpu_tensor, sizeof(char), buf_size, fp);
  fclose(fp);
  return allspark::AsStatus::ALLSPARK_SUCCESS;
}
#if 0
void test_model(const char* model_name, AsModelConfig as_model_config,
                const DLTensorMap& inputs, DLTensorMap* outputs) {
    allspark::AsEngine as_engine;
    AS_CHECK(as_engine.BuildModelFromConfigStruct(as_model_config));
    // TODO: adapt v2 API
    // AS_CHECK(as_engine.RunModel(model_name, inputs, outputs));
}
void test_model(const char* model_name, const char* device_type,
                const std::string& config_file, const DLTensorMap& inputs,
                DLTensorMap* outputs) {
    throw std::runtime_error("v1 API deprecated");
// deprecated
#if 0
    allspark::AsEngine as_engine;
    AS_CHECK(as_engine.SetDeviceType(device_type));
    std::string config_path(config_file);
    AS_CHECK(as_engine.BuildModelFromConfig(model_name, config_path.c_str()));
    AS_CHECK(as_engine.RunModel(model_name, inputs, outputs));
#endif
}
void test_model_generation(const char* model_name,
                           AsModelConfig as_model_config,
                           const DLTensorMap& inputs, DLTensorMap* outputs,
                           allspark::GenerateConfig& gen_cfg) {
    allspark::AsEngine as_engine;
    AS_CHECK(as_engine.BuildModelFromConfigStruct(as_model_config));
    // TODO: adapt v2 API
    // AS_CHECK(as_engine.RunTextGeneration(model_name, inputs, outputs,
    // gen_cfg));
}
void test_model_generation(const char* model_name, const char* device_type,
                           const std::string& config_file,
                           const DLTensorMap& inputs, DLTensorMap* outputs,
                           allspark::GenerateConfig& gen_cfg) {
    throw std::runtime_error("v1 API deprecated");
// deprecated
#if 0
    allspark::AsEngine as_engine;
    AS_CHECK(as_engine.SetDeviceType(device_type));
    std::string config_path(config_file);
    AS_CHECK(as_engine.BuildModelFromConfig(model_name, config_path.c_str()));
    AS_CHECK(as_engine.RunTextGeneration(model_name, inputs, outputs, gen_cfg));
#endif
}
#endif

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
