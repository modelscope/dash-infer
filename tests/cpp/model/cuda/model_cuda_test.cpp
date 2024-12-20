/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model_cuda_test.cpp
 */

#include <core/tensor/tensor.h>
#include <device_context.h>
#include <test_common.h>
#include <utility/timer.h>

#include <future>
#include <thread>

// #define CUDA_DEVICE "CUDA:0"
#define CUDA_DEVICE "CUDA:0,1"

template <typename T>
void extendVector(std::vector<T>& vec, int times) {
  std::vector<T> original = vec;    // 复制原始向量
  vec.reserve(vec.size() * times);  // 提前分配足够的空间
  for (int i = 1; i < times; ++i) {
    vec.insert(vec.end(), original.begin(), original.end());
  }
}
using namespace allspark;
class AsModelCUDA : public ::testing::Test {
 protected:
  void SetUp() override {
    device_context = allspark::DeviceContextFactory::CreateCUDAContext();
    qwen1_7b_test1_input_tokens_long = {
        151644, 872,    198,    108965, 11622,  109427, 9370,   99689,
        110818, 99842,  38035,  101651, 99500,  116481, 3837,   103962,
        99491,  100825, 101651, 3837,   115698, 100397, 107909, 100380,
        3837,   73670,  98237,  99534,  101883, 1773,   100431, 110068,
        105966, 108704, 5122,   80443,  103929, 100651, 73345,  80158,
        22045,  34187,  151645, 198,    151644, 77091};

    extendVector(qwen1_7b_test1_input_tokens_long, 80);  // 7k

    qwen1_7b_test1_input_tokens = {
        151644, 872,    198,    108965, 11622,  109427, 9370,   99689,
        110818, 99842,  38035,  101651, 99500,  116481, 3837,   103962,
        99491,  100825, 101651, 3837,   115698, 100397, 107909, 100380,
        3837,   73670,  98237,  99534,  101883, 1773,   100431, 110068,
        105966, 108704, 5122,   80443,  103929, 100651, 73345,  80158,
        22045,  34187,  151645, 198,    151644, 77091};

    qwen1_7b_test1_output_tokens = {
        151644, 872,    198,    108965, 11622,  109427, 9370,   99689,  110818,
        99842,  38035,  101651, 99500,  116481, 3837,   103962, 99491,  100825,
        101651, 3837,   115698, 100397, 107909, 100380, 3837,   73670,  98237,
        99534,  101883, 1773,   100431, 110068, 105966, 108704, 5122,   80443,
        103929, 100651, 73345,  80158,  22045,  34187,  151645, 198,    151644,
        77091,  271,    108668, 3837,   43288,  101228, 46944,  104098, 116305,
        9370,   102174, 6313,   103929, 100651, 17714,  73345,  106248, 34187,
        42192,  99739,  105774, 47534,  33108,  104523, 3837,   104092, 105361,
        116106, 9370,   109014, 116176, 34187,  101908, 105339, 1773,   103929,
        112283, 33108,  102379, 99258,  101908, 103932, 106890, 99880,  33108,
        100738, 3837,   104233, 100698, 105745, 102340, 60548,  108530, 1773,
        107427, 103929, 100651, 3837,   73345,  101244, 36993,  22045,  3837,
        111207, 104305, 103929, 102007, 3837,   104305, 56568,  99258,  103952,
        102294, 104286, 101163, 1773,   105043, 97639,  103932, 109153, 113013,
        108385, 3837,   97639,  102099, 101938, 112028, 103929, 47606,  1773,
        151645, 198,    151643};
  }
  // void TearDown() override {}

  std::shared_ptr<allspark::DeviceContext> device_context;
  std::vector<int64_t> qwen1_7b_test1_input_tokens;
  std::vector<int64_t> qwen1_7b_test1_input_tokens_long;

  std::vector<int64_t> qwen1_7b_test1_output_tokens;

  void test_model_swap_in_out(const char* model_name,
                              AsModelConfig as_model_config,
                              const DLTensorMap& inputs, DLTensorMap* outputs,
                              allspark::GenerateConfig& gen_cfg) {
    allspark::AsEngine as_engine;
    as_model_config.swap_threshold = 1;
    AS_CHECK(as_engine.BuildModelFromConfigStruct(as_model_config));

    ASSERT_EQ(as_engine.StartModel(model_name),
              allspark::AsStatus::ALLSPARK_SUCCESS);

// TODO: adapt v2 API
#if 0
        // expect success
        auto ret =
            as_engine.RunTextGeneration(model_name, inputs, outputs, gen_cfg);

        EXPECT_EQ(ret, AsStatus::ALLSPARK_SUCCESS);
        as_engine.UnloadModelFromDeviceMemory(model_name);

        // expect have exception.
        auto status =
            as_engine.RunTextGeneration(model_name, inputs, outputs, gen_cfg);
        EXPECT_EQ(status, AsStatus::ALLSPARK_PARAM_ERROR);

        as_engine.ReloadModelToDeviceMemory(model_name);
        // expect success
        as_engine.RunTextGeneration(model_name, inputs, outputs, gen_cfg);
#endif
  }

  void fetch_batch_request_by_multiple_thread(
      std::vector<std::vector<int64_t>>& fetched_tokens,
      const std::vector<RequestHandle_t>& pending_handles,
      const std::vector<AsEngine::ResultQueue_t>& pending_queue,
      std::vector<allspark::AsEngine::GenerateRequestStatus>& final_state,
      bool use_no_wait) {
    for (int i = 0; i < fetched_tokens.size(); i++) {
      auto& token_vec = fetched_tokens[i];
      auto& handle = pending_handles[i];
      auto& queue = pending_queue[i];
      auto& state = final_state[i];

      auto result_from_waitting = std::async(
          std::launch::async,
          [i, &state](RequestHandle_t handle, AsEngine::ResultQueue_t queue,
                      bool use_no_wait) -> std::vector<int64_t> {
            auto ret = std::vector<int64_t>();
            while (true) {
              auto status = queue->GenerateStatus();
              if (status != allspark::AsEngine::GenerateRequestStatus::
                                GenerateFinished &&
                  status != allspark::AsEngine::GenerateRequestStatus::
                                GenerateInterrupted) {
                if (use_no_wait) {
                  auto ele = queue->GetNoWait();
                  if (ele && ele->ids_from_generate.size() > 0) {
                    std::copy(ele->ids_from_generate.begin(),
                              ele->ids_from_generate.end(),
                              std::back_inserter(ret));
                  }
                  usleep(10000);
                  std::this_thread::yield();
                } else {
                  auto ele = queue->Get();

                  if (ele && ele->ids_from_generate.size() > 0) {
                    std::copy(ele->ids_from_generate.begin(),
                              ele->ids_from_generate.end(),
                              std::back_inserter(ret));
                  }
                }
              } else {
                if (status == allspark::AsEngine::GenerateRequestStatus::
                                  GenerateFinished ||
                    status == allspark::AsEngine::GenerateRequestStatus::
                                  GenerateInterrupted) {
                  // even finished, just try to get last message.
                  auto ele = queue->Get();

                  if (ele && ele->ids_from_generate.size() > 0) {
                    std::copy(ele->ids_from_generate.begin(),
                              ele->ids_from_generate.end(),
                              std::back_inserter(ret));
                    printf("generate status become finish. return value \n");
                  }
                  state = status;
                  break;
                } else {
                  printf("status invalid : %d \n", (int)status);
                }
              }
            }
            return ret;
          },
          handle, queue, use_no_wait);

      fetched_tokens[i] = result_from_waitting.get();
    }
  }
};

void PrintLogQueue(std::vector<AsEngine::ResultQueue_t>& pending_queue) {
  LOG(INFO) << "PrintLogQueue ";
  for (auto q_ptr : pending_queue) {
    if (!q_ptr) continue;

    auto ele = q_ptr->Get();
    if (!ele) continue;

    LOG(INFO) << "ptr: " << q_ptr << "ele ptr: " << ele.get()
              << " ele size: " << ele->ids_from_generate.size()
              << " status: " << q_ptr->GenerateStatus();

    for (int i = 0; i < ele->ids_from_generate.size(); i++) {
      LOG(INFO) << "ele[" << i << "]: " << ele->ids_from_generate[i];
    }
  }
}

size_t SumTotalToken(std::vector<AsEngine::ResultQueue_t>& pending_queue) {
  size_t sum = 0;
  for (auto q_ptr : pending_queue) {
    if (!q_ptr) continue;
    auto ele = q_ptr->Get();
    if (!ele) continue;
    sum += ele->ids_from_generate.size();
  }
  return sum;
}

TEST_F(AsModelCUDA, M6_7B_CacheDefault_Interrupted) {
  const std::string model_name = "m6_7b";
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 1;
  constexpr int max_batch_in_test = 10;

  const std::vector<int64_t> in0_data = qwen1_7b_test1_input_tokens_long;
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(seq_len, 1);

  auto out1_data = qwen1_7b_test1_output_tokens;
  const int out_len = static_cast<int>(out1_data.size());

  allspark::AsTensor in0("input_ids", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({1, seq_len}));
  allspark::AsTensor in1("attention_mask", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({1, seq_len}));

  in0.CopyDataFrom(in0_data.data(), 1 * seq_len * sizeof(int64_t),
                   allspark::CPU);
  in1.CopyDataFrom(in1_data.data(), 1 * seq_len * sizeof(int64_t),
                   allspark::CPU);
  const DLTensorMap inputs = {
      {"input_ids", in0.ToDLPack(device_context.get())}};

  std::string graph_path = model_path + "/" + model_name + ".asgraph";
  std::string weight_path = model_path + "/" + model_name + ".asparam";

  AsModelConfig as_model_config = AsModelConfig(
      model_name, graph_path.c_str(), weight_path.c_str(), CUDA_DEVICE);

  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length =
      qwen1_7b_test1_input_tokens_long.size() + 1024;
  // as_model_config.cache_span_size = 32;
  as_model_config.cache_mode = AsCacheMode::AsCacheDefault;
  as_model_config.enable_prefix_cache =
      false;  //  disable cache to make more interrupt.

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = as_model_config.engine_max_length;
    cfg->early_stopping = false;
    cfg->top_k = 0;
    cfg->top_p = 0.1;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;

  auto file_version_info =
      as_engine.GetFileInformation(graph_path.c_str(), weight_path.c_str());

  ASSERT_EQ(file_version_info.create_version_graph, "2.0.0");
  ASSERT_EQ(file_version_info.create_version_param, "2.0.0");

  ASSERT_EQ(as_engine.BuildModelFromConfigStruct(as_model_config),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.StartModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  std::vector<std::shared_ptr<AsEngine::RequestContent>> reqs(
      max_batch_in_test);
  std::vector<RequestHandle_t> pending_handles(max_batch_in_test);
  std::vector<AsEngine::ResultQueue_t> pending_queue(max_batch_in_test);

  for (int i = 0; i < max_batch_in_test; ++i) {
    std::shared_ptr<AsEngine::RequestContent> req =
        std::make_shared<AsEngine::RequestContent>();
    req->config = *(gen_config_vec[i]);
    req->infer_type = AsEngine::RequestInferType::Generate;
    req->inputs = std::make_shared<DLTensorMap>(inputs);
    req->mm_type = AsEngine::RequestMMType::TextInput;
    req->config.early_stopping = false;
    reqs[i] = std::move(req);
  }

  util::Timer timer;
  std::vector<std::future<AsStatus>> batch_result(max_batch_in_test);

  for (int wave = 0; wave < num_waves; ++wave) {
    LOG(INFO) << "================================================="
              << "Wave " << wave
              << "================================================="
              << std::endl;

    auto time_start = timer.elapsed();

    // request wave 1
    for (int i = 0; i < max_batch_in_test; ++i) {
      LOG(INFO) << "Wave " << wave << " test.start request: " << i;

      batch_result[i] = std::async(
          std::launch::async, [&, i, model_name]() -> allspark::AsStatus {
            auto status = as_engine.StartRequest(model_name.c_str(), reqs[i],
                                                 &(pending_handles[i]),
                                                 &(pending_queue[i]));
            printf(" request %d return status:%d\n", i, int(status));
            return status;
          });
    }

    sleep(30);
    std::vector<bool> status_vect(batch_result.size());
    for (int i = 0; i < max_batch_in_test; ++i) {
      ASSERT_EQ(batch_result[i].get(), allspark::AsStatus::ALLSPARK_SUCCESS);
      // even token may not enough token, it still will return sucecss
    }
    std::vector<std::vector<int64_t>> fetched_tokens(max_batch_in_test);
    std::vector<allspark::AsEngine::GenerateRequestStatus> request_final_states(
        max_batch_in_test);

    // use async fetch request.
    fetch_batch_request_by_multiple_thread(fetched_tokens, pending_handles,
                                           pending_queue, request_final_states,
                                           wave % 1);

    // sync all
    for (auto& handle : pending_handles) {
      ASSERT_EQ(as_engine.SyncRequest(model_name.c_str(), handle),
                allspark::AsStatus::ALLSPARK_SUCCESS);
    }
    int success_cnt = 0;
    int interrupt_cnt = 0;
    int error_cnt = 0;
    for (auto& state : request_final_states) {
      if (state ==
          allspark::AsEngine::GenerateRequestStatus::GenerateFinished) {
        success_cnt++;
        printf("generate finished. \n");
      } else if (state == allspark::AsEngine::GenerateRequestStatus::
                              GenerateInterrupted) {
        interrupt_cnt++;
        printf(" generate interrupted \n");
      } else if (state ==
                 allspark::AsEngine::GenerateRequestStatus::InternalError) {
        error_cnt++;
        printf("generate internal error\n");
      }
    }
    ASSERT_GT(interrupt_cnt, 1);
    ASSERT_GT(success_cnt, 1);
    ASSERT_EQ(interrupt_cnt + success_cnt, max_batch_in_test);

    for (auto& handle : pending_handles) {
      ASSERT_EQ(as_engine.StopRequest(model_name.c_str(), handle),
                allspark::AsStatus::ALLSPARK_SUCCESS);

      ASSERT_EQ(as_engine.ReleaseRequest(model_name.c_str(), handle),
                allspark::AsStatus::ALLSPARK_SUCCESS);
    }
  }

  LOG(INFO) << "================================================="
            << "Stop Model"
            << "=================================================" << std::endl;

  // this is required to release the model loop thread
  ASSERT_EQ(as_engine.StopModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.ReleaseModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in0.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in1.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
}

TEST_F(AsModelCUDA, M6_7B_BS100) {
  const std::string model_name = "m6_7b";
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 2;
  constexpr int max_batch_in_test = 100;

  const std::vector<int64_t> in0_data = qwen1_7b_test1_input_tokens;
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(seq_len, 1);

  auto out1_data = qwen1_7b_test1_output_tokens;
  const int out_len = static_cast<int>(out1_data.size());

  allspark::AsTensor in0("input_ids", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({1, seq_len}));
  allspark::AsTensor in1("attention_mask", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({1, seq_len}));

  in0.CopyDataFrom(in0_data.data(), 1 * seq_len * sizeof(int64_t),
                   allspark::CPU);
  in1.CopyDataFrom(in1_data.data(), 1 * seq_len * sizeof(int64_t),
                   allspark::CPU);
  const DLTensorMap inputs = {
      {"input_ids", in0.ToDLPack(device_context.get())}};

  std::string graph_path = model_path + "/" + model_name + ".asgraph";
  std::string weight_path = model_path + "/" + model_name + ".asparam";

  AsModelConfig as_model_config = AsModelConfig(
      model_name, graph_path.c_str(), weight_path.c_str(), CUDA_DEVICE);

  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length = 1024;
  // as_model_config.cache_span_size = 32;
  as_model_config.cache_mode = AsCacheMode::AsCacheDefault;
  as_model_config.prefill_mode = AsMHAPrefill::AsPrefillXformer;
  as_model_config.enable_prefix_cache = false;

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test * num_waves; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = out_len;
    cfg->early_stopping = false;
    cfg->top_k = 0;
    cfg->top_p = 0.1;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;

  auto file_version_info =
      as_engine.GetFileInformation(graph_path.c_str(), weight_path.c_str());

  ASSERT_EQ(file_version_info.create_version_graph, "2.0.0");
  ASSERT_EQ(file_version_info.create_version_param, "2.0.0");

  ASSERT_EQ(as_engine.BuildModelFromConfigStruct(as_model_config),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.StartModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  std::vector<std::shared_ptr<AsEngine::RequestContent>> reqs(
      max_batch_in_test * num_waves);
  std::vector<RequestHandle_t> pending_handles(max_batch_in_test * num_waves);
  std::vector<AsEngine::ResultQueue_t> pending_queue(max_batch_in_test *
                                                     num_waves);

  for (int i = 0; i < max_batch_in_test * num_waves; ++i) {
    std::shared_ptr<AsEngine::RequestContent> req =
        std::make_shared<AsEngine::RequestContent>();
    req->config = *(gen_config_vec[i]);
    req->infer_type = AsEngine::RequestInferType::Generate;
    req->inputs = std::make_shared<DLTensorMap>(inputs);
    req->mm_type = AsEngine::RequestMMType::TextInput;
    reqs[i] = std::move(req);
  }

  util::Timer timer;
  std::vector<std::future<AsStatus>> batch_result(max_batch_in_test *
                                                  num_waves);

  LOG(INFO) << "================================================="
            << "Wave " << 0
            << "=================================================" << std::endl;

  auto time_start = timer.elapsed();

  // request wave 1
  for (int i = 0; i < max_batch_in_test * num_waves; ++i) {
    LOG(INFO) << "Wave " << 0 << " test.start request: " << i;

    batch_result[i] = std::async(
        std::launch::async, [&, i, model_name]() -> allspark::AsStatus {
          return as_engine.StartRequest(model_name.c_str(), reqs[i],
                                        &(pending_handles[i]),
                                        &(pending_queue[i]));
        });
  }

  for (int i = 0; i < max_batch_in_test * num_waves; ++i) {
    EXPECT_EQ(batch_result[i].get(), allspark::AsStatus::ALLSPARK_SUCCESS);
    LOG(INFO) << "Wave " << 0 << " test.start, finish request: " << i;
  }

  std::vector<std::vector<int64_t>> fetched_tokens(max_batch_in_test *
                                                   num_waves);

  std::vector<allspark::AsEngine::GenerateRequestStatus> final_state(
      max_batch_in_test * num_waves);
  // use async fetch request.
  fetch_batch_request_by_multiple_thread(fetched_tokens, pending_handles,
                                         pending_queue, final_state, 0);

  // sync all
  ASSERT_EQ(as_engine.SyncRequest(model_name.c_str(), nullptr),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  auto time_end = timer.elapsed();
  auto duration = time_end - time_start;

  size_t sum = 0;
  for (int i = 0; i < fetched_tokens.size(); i++) {
    sum += fetched_tokens[i].size();

    std::vector<int64_t> result = in0_data;
    for (int j = 0; j < fetched_tokens[i].size(); j++)
      result.push_back(fetched_tokens[i][j]);
    float eps1 = check_equal_vec<int64_t>(result, out1_data, true);
    EXPECT_LE(eps1, MODEL_EPS);
  }

  int total_count = sum;
  LOG(INFO) << "Wave " << 0 << " Total Tokens  " << total_count
            << " ms: " << duration
            << " throughput: " << (total_count) / (duration / 1000.0f);

  for (auto& handle : pending_handles) {
    ASSERT_EQ(as_engine.ReleaseRequest(model_name.c_str(), handle),
              allspark::AsStatus::ALLSPARK_SUCCESS);
    static int idx = 0;
    LOG(INFO) << "Wave " << 0 << " test.release: " << idx++;
  }

  LOG(INFO) << "================================================="
            << "Stop Model"
            << "=================================================" << std::endl;

  // this is required to release the model loop thread
  ASSERT_EQ(as_engine.StopModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.ReleaseModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in0.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in1.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
}

TEST_F(AsModelCUDA, M6_7B_CacheDefault) {
  const std::string model_name = "m6_7b";
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 1;
  constexpr int max_batch_in_test = 2;

  const std::vector<int64_t> in0_data = qwen1_7b_test1_input_tokens;
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(seq_len, 1);

  auto out1_data = qwen1_7b_test1_output_tokens;
  const int out_len = static_cast<int>(out1_data.size());

  allspark::AsTensor in0("input_ids", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({1, seq_len}));
  allspark::AsTensor in1("attention_mask", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({1, seq_len}));

  in0.CopyDataFrom(in0_data.data(), 1 * seq_len * sizeof(int64_t),
                   allspark::CPU);
  in1.CopyDataFrom(in1_data.data(), 1 * seq_len * sizeof(int64_t),
                   allspark::CPU);
  const DLTensorMap inputs = {
      {"input_ids", in0.ToDLPack(device_context.get())}};

  std::string graph_path = model_path + "/" + model_name + ".asgraph";
  std::string weight_path = model_path + "/" + model_name + ".asparam";

  AsModelConfig as_model_config = AsModelConfig(
      model_name, graph_path.c_str(), weight_path.c_str(), CUDA_DEVICE);

  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length = 1024;
  // as_model_config.cache_span_size = 32;
  as_model_config.cache_mode = AsCacheMode::AsCacheDefault;
  as_model_config.prefill_mode = AsMHAPrefill::AsPrefillXformer;
  as_model_config.enable_prefix_cache = false;

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = out_len;
    cfg->early_stopping = false;
    cfg->top_k = 0;
    cfg->top_p = 0.1;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;

  auto file_version_info =
      as_engine.GetFileInformation(graph_path.c_str(), weight_path.c_str());

  ASSERT_EQ(file_version_info.create_version_graph, "2.0.0");
  ASSERT_EQ(file_version_info.create_version_param, "2.0.0");

  ASSERT_EQ(as_engine.BuildModelFromConfigStruct(as_model_config),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.StartModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  std::vector<std::shared_ptr<AsEngine::RequestContent>> reqs(
      max_batch_in_test);
  std::vector<RequestHandle_t> pending_handles(max_batch_in_test);
  std::vector<AsEngine::ResultQueue_t> pending_queue(max_batch_in_test);

  for (int i = 0; i < max_batch_in_test; ++i) {
    std::shared_ptr<AsEngine::RequestContent> req =
        std::make_shared<AsEngine::RequestContent>();
    req->config = *(gen_config_vec[i]);
    req->infer_type = AsEngine::RequestInferType::Generate;
    req->inputs = std::make_shared<DLTensorMap>(inputs);
    req->mm_type = AsEngine::RequestMMType::TextInput;
    reqs[i] = std::move(req);
  }

  util::Timer timer;
  std::vector<std::future<AsStatus>> batch_result(max_batch_in_test);

  for (int wave = 0; wave < num_waves; ++wave) {
    LOG(INFO) << "================================================="
              << "Wave " << wave
              << "================================================="
              << std::endl;

    auto time_start = timer.elapsed();

    // request wave 1
    for (int i = 0; i < max_batch_in_test; ++i) {
      LOG(INFO) << "Wave " << wave << " test.start request: " << i;

      batch_result[i] = std::async(
          std::launch::async, [&, i, model_name]() -> allspark::AsStatus {
            return as_engine.StartRequest(model_name.c_str(), reqs[i],
                                          &(pending_handles[i]),
                                          &(pending_queue[i]));
          });
    }

    for (int i = 0; i < max_batch_in_test; ++i) {
      EXPECT_EQ(batch_result[i].get(), allspark::AsStatus::ALLSPARK_SUCCESS);
      LOG(INFO) << "Wave " << wave << " test.start, finish request: " << i;
    }

    std::vector<std::vector<int64_t>> fetched_tokens(max_batch_in_test);

    std::vector<allspark::AsEngine::GenerateRequestStatus> final_state(
        max_batch_in_test);
    // use async fetch request.
    fetch_batch_request_by_multiple_thread(
        fetched_tokens, pending_handles, pending_queue, final_state, wave % 1);

    // sync all
    ASSERT_EQ(as_engine.SyncRequest(model_name.c_str(), nullptr),
              allspark::AsStatus::ALLSPARK_SUCCESS);

    auto time_end = timer.elapsed();
    auto duration = time_end - time_start;

    size_t sum = 0;
    for (int i = 0; i < fetched_tokens.size(); i++) {
      sum += fetched_tokens[i].size();

      std::vector<int64_t> result = in0_data;
      for (int j = 0; j < fetched_tokens[i].size(); j++)
        result.push_back(fetched_tokens[i][j]);
      float eps1 = check_equal_vec<int64_t>(result, out1_data, true);
      EXPECT_LE(eps1, MODEL_EPS);
    }

    int total_count = sum;
    LOG(INFO) << "Wave " << wave << " Total Tokens  " << total_count
              << " ms: " << duration
              << " throughput: " << (total_count) / (duration / 1000.0f);

    for (auto& handle : pending_handles) {
      ASSERT_EQ(as_engine.ReleaseRequest(model_name.c_str(), handle),
                allspark::AsStatus::ALLSPARK_SUCCESS);
      static int idx = 0;
      LOG(INFO) << "Wave " << wave << " test.release: " << idx++;
    }
  }

  LOG(INFO) << "================================================="
            << "Stop Model"
            << "=================================================" << std::endl;

  // this is required to release the model loop thread
  ASSERT_EQ(as_engine.StopModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.ReleaseModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in0.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in1.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
}

TEST_F(AsModelCUDA, M6_7B_CacheI8) {
  const std::string model_name = "m6_7b";
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 1;
  constexpr int max_batch_in_test = 2;
  constexpr int batch_size = 1;
  /*
   * <|im_start|>user\n用一句话介绍一下大语言模型的量化技术。<|im_end|>\n<|im_start|>assistant
   */
  const std::vector<int64_t> in0_data = {
      151644, 872,    198,   11622, 105321, 109432, 26288,  102064, 104949,
      9370,   108048, 99361, 1773,  151645, 198,    151644, 77091};
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(batch_size * seq_len, 1);

  /*
   * <|im_start|>user\n用一句话介绍一下大语言模型的量化技术。<|im_end|>\n<|im_start|>assistant
   * 大语言模型的量化技术是指通过数学模型和统计方法对大语言模型的参数进行建模和优化，
   * 从而实现对大语言模型的高效计算和应用。<|im_end|>\n<|endoftext|>
   */
  const std::vector<int64_t> out1_data = {
      151644, 872,    198,    11622,  105321, 109432, 26288,  102064, 104949,
      9370,   108048, 99361,  1773,   151645, 198,    151644, 77091,  26288,
      102064, 104949, 9370,   108048, 99361,  104442, 67338,  104552, 104949,
      33108,  100787, 39907,  32664,  26288,  102064, 104949, 9370,   32665,
      71817,  25807,  53772,  33108,  103983, 3837,   101982, 101884, 32664,
      26288,  102064, 104949, 9370,   102202, 100768, 33108,  99892,  1773,
      151645, 198,    151643};
  const int out_len = static_cast<int>(out1_data.size());

  allspark::AsTensor in0("input_ids", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));
  allspark::AsTensor in1("attention_mask", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));

  in0.CopyDataFrom(in0_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  in1.CopyDataFrom(in1_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  const DLTensorMap inputs = {
      {"input_ids", in0.ToDLPack(device_context.get())},
      {"attention_mask", in1.ToDLPack(device_context.get())}};

  AsModelConfig as_model_config =
      AsModelConfig(model_name, model_path + "/" + model_name + ".asgraph",
                    model_path + "/" + model_name + ".asparam", CUDA_DEVICE);
  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length = 1024;
  // as_model_config.cache_span_size = 32;
  as_model_config.cache_mode = AsCacheMode::AsCacheQuantI8;
  as_model_config.prefill_mode = AsMHAPrefill::AsPrefillXformer;
  as_model_config.enable_prefix_cache = false;

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = out_len;
    cfg->early_stopping = false;
    cfg->top_k = 0;
    cfg->top_p = 0.1;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;
  ASSERT_EQ(as_engine.BuildModelFromConfigStruct(as_model_config),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.StartModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  std::vector<std::shared_ptr<AsEngine::RequestContent>> reqs(
      max_batch_in_test);
  std::vector<RequestHandle_t> pending_handles(max_batch_in_test);
  std::vector<AsEngine::ResultQueue_t> pending_queue(max_batch_in_test);

  for (int i = 0; i < max_batch_in_test; ++i) {
    std::shared_ptr<AsEngine::RequestContent> req =
        std::make_shared<AsEngine::RequestContent>();
    req->config = *(gen_config_vec[i]);
    req->infer_type = AsEngine::RequestInferType::Generate;
    req->inputs = std::make_shared<DLTensorMap>(inputs);
    req->mm_type = AsEngine::RequestMMType::TextInput;
    reqs[i] = std::move(req);
  }

  util::Timer timer;
  std::vector<std::future<AsStatus>> result(max_batch_in_test);

  for (int wave = 0; wave < num_waves; ++wave) {
    LOG(INFO) << "================================================="
              << "Wave " << wave
              << "================================================="
              << std::endl;

    auto time_start = timer.elapsed();

    // request wave 1
    for (int i = 0; i < max_batch_in_test; ++i) {
      LOG(INFO) << "Wave " << wave << " test.start request: " << i;

      result[i] = std::async(
          std::launch::async, [&, i, model_name]() -> allspark::AsStatus {
            return as_engine.StartRequest(model_name.c_str(), reqs[i],
                                          &(pending_handles[i]),
                                          &(pending_queue[i]));
          });
    }

    for (int i = 0; i < max_batch_in_test; ++i) {
      EXPECT_EQ(result[i].get(), allspark::AsStatus::ALLSPARK_SUCCESS);
      LOG(INFO) << "Wave " << wave << " test.start, finish request: " << i;
    }

    // sync all
    ASSERT_EQ(as_engine.SyncRequest(model_name.c_str(), nullptr),
              allspark::AsStatus::ALLSPARK_SUCCESS);

    auto time_end = timer.elapsed();
    auto duration = time_end - time_start;

    size_t sum = 0;
    for (auto q_ptr : pending_queue) {
      if (!q_ptr) continue;
      auto ele = q_ptr->Get();
      if (!ele) continue;

      sum += ele->ids_from_generate.size();
      std::vector<int64_t> result = in0_data;
      result.insert(result.end(),
                    std::move_iterator(ele->ids_from_generate.begin()),
                    std::move_iterator(ele->ids_from_generate.end()));

      int64_t eps1 = check_equal<int64_t>(result.data(), out1_data.data(),
                                          batch_size * out_len, true);
      EXPECT_LE(eps1, MODEL_EPS);
    }
    int total_count = sum;
    LOG(INFO) << "Wave " << wave << " Total Tokens  " << total_count
              << " ms: " << duration
              << " throughput: " << (total_count) / (duration / 1000.0f);

    for (auto& handle : pending_handles) {
      ASSERT_EQ(as_engine.ReleaseRequest(model_name.c_str(), handle),
                allspark::AsStatus::ALLSPARK_SUCCESS);
      static int idx = 0;
      LOG(INFO) << "Wave " << wave << " test.release: " << idx++;
    }
  }

  LOG(INFO) << "================================================="
            << "Stop Model"
            << "=================================================" << std::endl;

  // this is required to release the model loop thread
  ASSERT_EQ(as_engine.StopModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in0.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in1.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
}

TEST_F(AsModelCUDA, M6_7B_CacheU4) {
  const std::string model_name = "m6_7b";
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 1;
  constexpr int max_batch_in_test = 2;
  constexpr int batch_size = 1;
  constexpr int test_round = 1;
  /*
   * <|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant
   */
  const std::vector<int64_t> in0_data = {151644, 872,    198, 105043, 100165,
                                         11319,  151645, 198, 151644, 77091};
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(batch_size * seq_len, 1);

  /*
   * <|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant
   * 我是阿里巴巴达摩院开发的一款超大规模语言模型，我叫通义千问。<|im_end|>
   */
  const std::vector<int64_t> out1_data = {
      151644, 872,   198,    105043, 100165, 11319,  151645, 198,
      151644, 77091, 104198, 107076, 93488,  100487, 93823,  100013,
      110659, 71304, 105483, 102064, 104949, 3837,   35946,  99882,
      31935,  64559, 99320,  56007,  1773,   151645};
  const int out_len = static_cast<int>(out1_data.size());

  AsModelConfig as_model_config =
      AsModelConfig(model_name, model_path + "/" + model_name + ".asgraph",
                    model_path + "/" + model_name + ".asparam", CUDA_DEVICE);
  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length = 1024;
  // as_model_config.cache_span_size = 32;
  as_model_config.cache_mode = AsCacheMode::AsCacheQuantU4;
  as_model_config.prefill_mode = AsMHAPrefill::AsPrefillXformer;

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = out_len;
    cfg->early_stopping = false;
    cfg->top_k = 1;
    cfg->top_p = 0;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;
  ASSERT_EQ(as_engine.BuildModelFromConfigStruct(as_model_config),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.StartModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  std::vector<std::shared_ptr<AsEngine::RequestContent>> reqs(
      max_batch_in_test);
  std::vector<RequestHandle_t> pending_handles(max_batch_in_test);
  std::vector<AsEngine::ResultQueue_t> pending_queue(max_batch_in_test);

  // for (int cnt = 1; cnt < test_round; cnt++) {
  allspark::AsTensor in0("input_ids", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));
  allspark::AsTensor in1("attention_mask", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));

  in0.CopyDataFrom(in0_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  in1.CopyDataFrom(in1_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  const DLTensorMap inputs = {
      {"input_ids", in0.ToDLPack(device_context.get())},
      {"attention_mask", in1.ToDLPack(device_context.get())}};

  for (int i = 0; i < max_batch_in_test; ++i) {
    std::shared_ptr<AsEngine::RequestContent> req =
        std::make_shared<AsEngine::RequestContent>();
    req->config = *(gen_config_vec[i]);
    req->infer_type = AsEngine::RequestInferType::Generate;
    req->inputs = std::make_shared<DLTensorMap>(inputs);
    req->mm_type = AsEngine::RequestMMType::TextInput;
    reqs[i] = std::move(req);
  }

  util::Timer timer;
  std::vector<std::future<AsStatus>> result(max_batch_in_test);

  for (int wave = 0; wave < num_waves; ++wave) {
    LOG(INFO) << "================================================="
              << "Wave " << wave
              << "================================================="
              << std::endl;

    auto time_start = timer.elapsed();

    int test_count = 1;
    for (int cnt = 0; cnt < test_count; cnt++) {
      // request wave 1
      for (int i = 0; i < max_batch_in_test; ++i) {
        LOG(INFO) << "Wave " << wave << " test.start request: " << i;

        result[i] = std::async(
            std::launch::async, [&, i, model_name]() -> allspark::AsStatus {
              return as_engine.StartRequest(model_name.c_str(), reqs[i],
                                            &(pending_handles[i]),
                                            &(pending_queue[i]));
            });
      }

      for (int i = 0; i < max_batch_in_test; ++i) {
        EXPECT_EQ(result[i].get(), allspark::AsStatus::ALLSPARK_SUCCESS);
        LOG(INFO) << "Wave " << wave << " test.start, finish request: " << i;
      }

      // sync all
      ASSERT_EQ(as_engine.SyncRequest(model_name.c_str(), nullptr),
                allspark::AsStatus::ALLSPARK_SUCCESS);

      auto time_end = timer.elapsed();
      auto duration = time_end - time_start;

      size_t sum = 0;
      for (auto q_ptr : pending_queue) {
        if (!q_ptr) continue;
        auto ele = q_ptr->Get();
        if (!ele) continue;

        sum += ele->ids_from_generate.size();
        std::vector<int64_t> result = in0_data;
        result.insert(result.end(),
                      std::move_iterator(ele->ids_from_generate.begin()),
                      std::move_iterator(ele->ids_from_generate.end()));

        int64_t eps1 = check_equal<int64_t>(result.data(), out1_data.data(),
                                            batch_size * out_len, true);
        EXPECT_LE(eps1, MODEL_EPS);
      }
      int total_count = sum;
      LOG(INFO) << "Wave " << wave << " Total Tokens  " << total_count
                << " ms: " << duration
                << " throughput: " << (total_count) / (duration / 1000.0f);

      for (auto& handle : pending_handles) {
        ASSERT_EQ(as_engine.ReleaseRequest(model_name.c_str(), handle),
                  allspark::AsStatus::ALLSPARK_SUCCESS);
        static int idx = 0;
        LOG(INFO) << "Wave " << wave << " test.release: " << idx++;
      }
    }
  }

  ASSERT_EQ(in0.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in1.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  //}

  LOG(INFO) << "================================================="
            << "Stop Model"
            << "=================================================" << std::endl;

  // this is required to release the model loop thread
  ASSERT_EQ(as_engine.StopModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
}

TEST_F(AsModelCUDA, M6_7B_RichEmbedding) {
  const std::string model_name = "m6_7b_rich";
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 1;
  constexpr int max_batch_in_test = 2;
  constexpr int batch_size = 1;
  /*
  <fim_middle>system
  You are a helpful assistant.<fim_suffix>
  <fim_middle>user
  <extra_188><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_190><extra_189>这张图里面是什么<fim_suffix>
  <fim_middle>assistant
  */
  const std::vector<int64_t> in0_data = {
      151644, 8948,   198,    2610,   525,    264,    10950,  17847,  13,
      151645, 198,    151644, 872,    198,    151857, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151858, 108893, 28029,  100420, 102021, 151645, 198,    151644,
      77091,  198};
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(batch_size * seq_len, 1);

  /*
   这张图中是爱因斯坦的头像。<|im_end|>
  */
  const std::vector<int64_t> out1_data = {
      151644, 8948,   198,    2610,   525,    264,    10950,  17847,  13,
      151645, 198,    151644, 872,    198,    151857, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859,
      151859, 151858, 108893, 28029,  100420, 102021, 151645, 198,    151644,
      77091,  198,    108893, 28029,  15946,  20412,  99242,  62112,  103195,
      9370,   64355,  65101,  1773,   151645};

  std::vector<float> embedding_data(256 * 4096);
  AS_CHECK(LoadBinFromFile(model_path + "emb.bin", embedding_data.data(),
                           256 * 4096 * 4));
  const int out_len = static_cast<int>(out1_data.size());
  allspark::AsTensor in0("input_ids", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));
  allspark::AsTensor in1("attention_mask", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));
  allspark::AsTensor rich_embed0("rich_embedding", allspark::CPU,
                                 allspark::FLOAT32, allspark::DataMode::DENSE,
                                 allspark::Shape({256, 4096}));
  in0.CopyDataFrom(in0_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  in1.CopyDataFrom(in1_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  rich_embed0.CopyDataFrom(embedding_data.data(), 256 * 4096 * sizeof(float),
                           allspark::CPU);
  const DLTensorMap inputs = {
      {"input_ids", in0.ToDLPack(device_context.get())},
      {"attention_mask", in1.ToDLPack(device_context.get())}};
  // rich_embedding start

  // DLManagedTensor* dltensor = rich_embed0.ToDLPack(device_context.get());
  std::vector<DLManagedTensor*> dl_list = {
      rich_embed0.ToDLPack(device_context.get())};
  // std::vector<DLManagedTensor*> dl_list = {
  //     };
  //  dl_list[0]=std::move(dltensor);
  //  std::vector<int64_t> embedding_shape = {256, 4096};
  //  std::vector<std::vector<float>> embed_data_list;
  //  std::vector<std::vector<int64_t>> embed_shape_list;
  //  embed_data_list.push_back(embedding_data);
  //  embed_shape_list.push_back(embedding_shape);

  MultiMediaInfo as_extra_embedding_info_0 = MultiMediaInfo();
  as_extra_embedding_info_0.set_multimedia_type(0);
  as_extra_embedding_info_0.add_multimedia_content(std::string("151859"),
                                                   dl_list);
  // rich_embedding over
  AsModelConfig as_model_config =
      AsModelConfig(model_name, model_path + "/" + model_name + ".asgraph",
                    model_path + "/" + model_name + ".asparam", CUDA_DEVICE);
  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length = 1024;
  // as_model_config.cache_span_size = 32;
  as_model_config.enable_prefix_cache = false;

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = out_len;
    cfg->early_stopping = false;
    cfg->top_k = 1;
    cfg->mm_info = &as_extra_embedding_info_0;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;
  EXPECT_EQ(as_engine.BuildModelFromConfigStruct(as_model_config),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  EXPECT_EQ(as_engine.StartModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  std::vector<std::shared_ptr<AsEngine::RequestContent>> reqs(
      max_batch_in_test);
  std::vector<RequestHandle_t> pending_handles(max_batch_in_test);
  std::vector<AsEngine::ResultQueue_t> pending_queue(max_batch_in_test);

  for (int i = 0; i < max_batch_in_test; ++i) {
    std::shared_ptr<AsEngine::RequestContent> req =
        std::make_shared<AsEngine::RequestContent>();
    req->config = *(gen_config_vec[i]);
    req->infer_type = AsEngine::RequestInferType::Generate;
    req->inputs = std::make_shared<DLTensorMap>(inputs);
    req->mm_type = AsEngine::RequestMMType::TextInput;
    reqs[i] = std::move(req);
  }

  util::Timer timer;
  std::vector<std::future<AsStatus>> result(max_batch_in_test);

  for (int wave = 0; wave < num_waves; ++wave) {
    LOG(INFO) << "================================================="
              << "Wave " << wave
              << "================================================="
              << std::endl;

    auto time_start = timer.elapsed();

    // request wave 1
    for (int i = 0; i < max_batch_in_test; ++i) {
      LOG(INFO) << "Wave " << wave << " test.start request: " << i;

      result[i] = std::async(
          std::launch::async, [&, i, model_name]() -> allspark::AsStatus {
            return as_engine.StartRequest(model_name.c_str(), reqs[i],
                                          &(pending_handles[i]),
                                          &(pending_queue[i]));
          });
    }

    for (int i = 0; i < max_batch_in_test; ++i) {
      EXPECT_EQ(result[i].get(), allspark::AsStatus::ALLSPARK_SUCCESS);
      LOG(INFO) << "Wave " << wave << " test.start, finish request: " << i;
    }

    // sync all
    EXPECT_EQ(as_engine.SyncRequest(model_name.c_str(), nullptr),
              allspark::AsStatus::ALLSPARK_SUCCESS);

    auto time_end = timer.elapsed();
    auto duration = time_end - time_start;

    size_t sum = 0;
    for (auto q_ptr : pending_queue) {
      if (!q_ptr) continue;
      auto ele = q_ptr->Get();
      if (!ele) continue;

      sum += ele->ids_from_generate.size();
      std::vector<int64_t> result = in0_data;
      result.insert(result.end(),
                    std::move_iterator(ele->ids_from_generate.begin()),
                    std::move_iterator(ele->ids_from_generate.end()));

      int64_t eps1 = check_equal<int64_t>(result.data(), out1_data.data(),
                                          batch_size * out_len, true);

      std::cout << "[";
      for (int i = 0; i < result.size(); i++) std::cout << result[i] << ",";
      std::cout << "]" << std::endl;
      EXPECT_LE(eps1, MODEL_EPS);
    }
    int total_count = sum;
    LOG(INFO) << "Wave " << wave << " Total Tokens  " << total_count
              << " ms: " << duration
              << " throughput: " << (total_count) / (duration / 1000.0f);

    for (auto& handle : pending_handles) {
      EXPECT_EQ(as_engine.ReleaseRequest(model_name.c_str(), handle),
                allspark::AsStatus::ALLSPARK_SUCCESS);
      static int idx = 0;
      LOG(INFO) << "Wave " << wave << " test.release: " << idx++;
    }
  }

  LOG(INFO) << "================================================="
            << "Stop Model"
            << "=================================================" << std::endl;

  // this is required to release the model loop thread
  EXPECT_EQ(as_engine.StopModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  EXPECT_EQ(in0.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  EXPECT_EQ(in1.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
}

TEST_F(AsModelCUDA, LLAMA_7B_ContinuousBatch) {
  std::string model_name = "llama2_7b";
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 1;
  constexpr int max_batch_in_test = 2;
  constexpr int batch_size = 1;
  /*
   * <s> What is the largest planet in our solar system?
   */
  const std::vector<int64_t> in0_data = {1,   1724, 338,   278,  10150, 15754,
                                         297, 1749, 21635, 1788, 29973};
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(batch_size * seq_len, 1);

  /*
   * <s> What is the largest planet in our solar system?
   * The largest planet in our solar system is Jupiter. It has a diameter of
   * approximately 89,000 miles (143,000 kilometers) and is more than 300
   * times more massive than Earth. Jupiter is a gas giant, meaning that it is
   * primarily composed of hydrogen and helium, and it has a thick atmosphere
   * with storm systems and a strong magnetic field.</s>
   */
  const std::vector<int64_t> out1_data = {
      1,     1724,  338,   278,   10150, 15754, 297,   1749,  21635, 1788,
      29973, 13,    13,    1576,  10150, 15754, 297,   1749,  21635, 1788,
      338,   27441, 1524,  29889, 739,   756,   263,   24235, 310,   14235,
      29871, 29947, 29929, 29892, 29900, 29900, 29900, 7800,  313,   29896,
      29946, 29941, 29892, 29900, 29900, 29900, 20052, 2699,  29897, 322,
      338,   901,   1135,  29871, 29941, 29900, 29900, 3064,  901,   20364,
      1135,  11563, 29889, 27441, 1524,  338,   263,   10489, 28396, 29892,
      6593,  393,   372,   338,   19434, 13725, 310,   17546, 1885,  322,
      1081,  1974,  29892, 322,   372,   756,   263,   12003, 25005, 411,
      14280, 6757,  322,   263,   4549,  15611, 1746,  29889, 2};

  const int out_len = static_cast<int>(out1_data.size());

  allspark::AsTensor in0("input_ids", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));
  allspark::AsTensor in1("attention_mask", allspark::CPU, allspark::INT64,
                         allspark::DataMode::DENSE,
                         allspark::Shape({batch_size, seq_len}));

  in0.CopyDataFrom(in0_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  in1.CopyDataFrom(in1_data.data(), batch_size * seq_len * sizeof(int64_t),
                   allspark::CPU);
  const DLTensorMap inputs = {
      {"input_ids", in0.ToDLPack(device_context.get())},
      {"attention_mask", in1.ToDLPack(device_context.get())}};

  std::string graph_path = model_path + "/" + model_name + ".asgraph";
  std::string weight_path = model_path + "/" + model_name + ".asparam";

  AsModelConfig as_model_config = AsModelConfig(
      model_name, graph_path.c_str(), weight_path.c_str(), CUDA_DEVICE);

  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length = 1024;
  // as_model_config.cache_span_size = 32;
  as_model_config.cache_mode = AsCacheMode::AsCacheDefault;
  as_model_config.prefill_mode = AsMHAPrefill::AsPrefillXformer;
  as_model_config.enable_prefix_cache = false;

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = out_len;
    cfg->early_stopping = false;
    cfg->top_k = 0;
    cfg->top_p = 0.1;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;

  auto file_version_info =
      as_engine.GetFileInformation(graph_path.c_str(), weight_path.c_str());

  ASSERT_EQ(file_version_info.create_version_graph, "2.1.0");
  ASSERT_EQ(file_version_info.create_version_param, "2.1.0");

  ASSERT_EQ(as_engine.BuildModelFromConfigStruct(as_model_config),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(as_engine.StartModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);

  std::vector<std::shared_ptr<AsEngine::RequestContent>> reqs(
      max_batch_in_test);
  std::vector<RequestHandle_t> pending_handles(max_batch_in_test);
  std::vector<AsEngine::ResultQueue_t> pending_queue(max_batch_in_test);

  for (int i = 0; i < max_batch_in_test; ++i) {
    std::shared_ptr<AsEngine::RequestContent> req =
        std::make_shared<AsEngine::RequestContent>();
    req->config = *(gen_config_vec[i]);
    req->infer_type = AsEngine::RequestInferType::Generate;
    req->inputs = std::make_shared<DLTensorMap>(inputs);
    req->mm_type = AsEngine::RequestMMType::TextInput;
    reqs[i] = std::move(req);
  }

  util::Timer timer;
  std::vector<std::future<AsStatus>> result(max_batch_in_test);

  for (int wave = 0; wave < num_waves; ++wave) {
    LOG(INFO) << "================================================="
              << "Wave " << wave
              << "================================================="
              << std::endl;

    auto time_start = timer.elapsed();

    // request wave 1
    for (int i = 0; i < max_batch_in_test; ++i) {
      LOG(INFO) << "Wave " << wave << " test.start request: " << i;

      result[i] = std::async(
          std::launch::async, [&, i, model_name]() -> allspark::AsStatus {
            return as_engine.StartRequest(model_name.c_str(), reqs[i],
                                          &(pending_handles[i]),
                                          &(pending_queue[i]));
          });
    }

    for (int i = 0; i < max_batch_in_test; ++i) {
      EXPECT_EQ(result[i].get(), allspark::AsStatus::ALLSPARK_SUCCESS);
      LOG(INFO) << "Wave " << wave << " test.start, finish request: " << i;
    }

    // sync all
    ASSERT_EQ(as_engine.SyncRequest(model_name.c_str(), nullptr),
              allspark::AsStatus::ALLSPARK_SUCCESS);

    auto time_end = timer.elapsed();
    auto duration = time_end - time_start;

    size_t sum = 0;
    for (auto q_ptr : pending_queue) {
      if (!q_ptr) continue;
      auto ele = q_ptr->Get();
      if (!ele) continue;

      sum += ele->ids_from_generate.size();
      std::vector<int64_t> result = in0_data;
      result.insert(result.end(),
                    std::move_iterator(ele->ids_from_generate.begin()),
                    std::move_iterator(ele->ids_from_generate.end()));

      int64_t eps1 = check_equal<int64_t>(result.data(), out1_data.data(),
                                          batch_size * out_len);
      EXPECT_LE(eps1, MODEL_EPS);
    }
    int total_count = sum;
    LOG(INFO) << "Wave " << wave << " Total Tokens  " << total_count
              << " ms: " << duration
              << " throughput: " << (total_count) / (duration / 1000.0f);

    for (auto& handle : pending_handles) {
      ASSERT_EQ(as_engine.ReleaseRequest(model_name.c_str(), handle),
                allspark::AsStatus::ALLSPARK_SUCCESS);
      static int idx = 0;
      LOG(INFO) << "Wave " << wave << " test.release: " << idx++;
    }
  }

  LOG(INFO) << "================================================="
            << "Stop Model"
            << "=================================================" << std::endl;

  // this is required to release the model loop thread
  ASSERT_EQ(as_engine.StopModel(model_name.c_str()),
            allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in0.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
  ASSERT_EQ(in1.Free(), allspark::AsStatus::ALLSPARK_SUCCESS);
}
