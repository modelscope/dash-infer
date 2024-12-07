/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model_cpu_test.cpp
 */

#include <core/tensor/tensor.h>
#include <device_context.h>
#include <test_common.h>
#include <utility/timer.h>

#include <future>
#include <thread>

class AsModelCPU : public ::testing::Test {
 protected:
  void SetUp() override {
    device_context = allspark::DeviceContextFactory::CreateCPUContext();
  }

  // void TearDown() override {}

  void test_model(std::string model_name, int batch_size,
                  const std::vector<int64_t>& in0_data,
                  const std::vector<int64_t>& in1_data,
                  const std::vector<int64_t>& out1_data,
                  std::string model_version);
  std::shared_ptr<allspark::DeviceContext> device_context;
};

void AsModelCPU::test_model(std::string model_name, int batch_size,
                            const std::vector<int64_t>& in0_data,
                            const std::vector<int64_t>& in1_data,
                            const std::vector<int64_t>& out1_data,
                            std::string model_version) {
  const std::string model_path = std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                                 "testcase/" + (model_name + "/");

  constexpr int num_waves = 1;
  constexpr int max_batch_in_test = 2;
  const int seq_len = static_cast<int>(in0_data.size());

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

  AsModelConfig as_model_config = AsModelConfig(model_name, graph_path.c_str(),
                                                weight_path.c_str(), "CPU:0");

  as_model_config.engine_max_batch = max_batch_in_test;
  as_model_config.engine_max_length = 1024;
  as_model_config.matmul_precision = "medium_bf16";
  as_model_config.prefill_mode = AsMHAPrefill::AsPrefillDefault;

  std::vector<std::unique_ptr<GenerateConfig>> gen_config_vec;
  for (int i = 0; i < max_batch_in_test; ++i) {
    auto cfg = std::make_unique<GenerateConfig>();
    cfg->max_length = out_len;
    cfg->early_stopping = false;
    cfg->top_k = 1024;
    cfg->top_p = 0.8;
    cfg->repetition_penalty = 1.1;
    gen_config_vec.emplace_back(std::move(cfg));
  }

  allspark::AsEngine as_engine;

  auto file_version_info =
      as_engine.GetFileInformation(graph_path.c_str(), weight_path.c_str());

  ASSERT_EQ(file_version_info.create_version_graph, model_version);
  ASSERT_EQ(file_version_info.create_version_param, model_version);

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

#if 0
            printf("reference data:\n");
            for (int i = 0; i < out1_data.size(); i++) {
                if (i == 0) printf("%lld", out1_data[i]);
                else printf(", %lld", out1_data[i]);
            }
            printf("\n");

            printf("result data:\n");
            for (int i = 0; i < result.size(); i++) {
                if (i == 0) printf("%lld", result[i]);
                else printf(", %lld", result[i]);
            }
            printf("\n");
#endif

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

#ifdef ENABLE_ARM_V84_V9
TEST_F(AsModelCPU, LLAMA_7B_ContinuousBatch) {
  std::string model_name = "llama2_7b_cpu";
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
   * The largest planet in our solar system is Jupiter. It has a diameter
   * of approximately 89,000 miles (143,000 kilometers) and is more than
   * 300 times more massive than Earth.</s>
   */
  const std::vector<int64_t> out1_data = {
      1,     1724,  338,   278,   10150, 15754, 297,   1749,  21635, 1788,
      29973, 13,    13,    1576,  10150, 15754, 297,   1749,  21635, 1788,
      338,   27441, 1524,  29889, 739,   756,   263,   24235, 310,   14235,
      29871, 29947, 29929, 29892, 29900, 29900, 29900, 7800,  313,   29896,
      29946, 29941, 29892, 29900, 29900, 29900, 20052, 2699,  29897, 322,
      338,   901,   1135,  29871, 29941, 29900, 29900, 3064,  901,   20364,
      1135,  11563, 29889, 2};

  std::string model_version = "2.1.0";
  test_model(model_name, batch_size, in0_data, in1_data, out1_data,
             model_version);
}

TEST_F(AsModelCPU, CHATGLM3_6B_32K_ContinuousBatch) {
  std::string model_name = "chatglm3_6b_32k_cpu_new";
  constexpr int batch_size = 1;

  /*
   * [gMASK]sop <|user|>
   * 讲解一下“温故而知新”<|assistant|>
   */
  const std::vector<int64_t> in0_data = {
      64790, 64792, 906,   31007, 4865,  31007, 30994, 13,
      33878, 32024, 30989, 55166, 55080, 41718, 54575, 30991,
      31002, 31007, 530,   18971, 31007, 30994};
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(batch_size * seq_len, 1);

  /*
   * [gMASK]sop <|user|>
   * 讲解一下“温故而知新”<|assistant|>
   * “温故而知新”是一句中国古代成语，出自《论语·为政》。它的意思是说通过回顾过去
   * 学过的知识，从而获得新的理解和认识。这句话强调了学习的重要性和持续性，告诉我
   * 们要不断巩固已学的知识，同时也要在新的环境中学习和探索。
   */
  const std::vector<int64_t> out1_data = {
      64790, 64792, 906,   31007, 4865,  31007, 30994, 13,    33878,
      32024, 30989, 55166, 55080, 41718, 54575, 30991, 31002, 31007,
      530,   18971, 31007, 30994, 13,    520,   55166, 55080, 41718,
      54575, 30991, 31690, 55390, 43354, 43471, 31123, 37808, 54611,
      47192, 31065, 54541, 54681, 32476, 32516, 43079, 54636, 31662,
      34331, 32151, 54545, 33258, 31848, 31123, 32583, 31823, 31888,
      46413, 32254, 31155, 34096, 48779, 31658, 32170, 33286, 31954,
      54642, 31123, 32345, 33883, 31800, 34776, 54757, 34818, 31848,
      31123, 31701, 33021, 52245, 42113, 39564, 32535, 31155, 2};

  std::string model_version = "3.2.0";
  test_model(model_name, batch_size, in0_data, in1_data, out1_data,
             model_version);
}

TEST_F(AsModelCPU, QWEN2_1_5B_ContinuousBatch) {
  std::string model_name = "qwen2_1_5b_cpu";
  constexpr int batch_size = 1;

  /*
   * <|im_start|>system
   * You are a helpful assistant.<|im_end|>
   * <|im_start|>user
   * 讲解一下“温故而知新”<|im_end|>
   * <|im_start|>assistant
   */
  const std::vector<int64_t> in0_data = {
      151644, 8948,   198, 2610,   525,    264,    10950, 17847, 13,    151645,
      198,    151644, 872, 198,    105250, 100158, 2073,  99416, 99535, 68536,
      52183,  16628,  854, 151645, 198,    151644, 77091, 198};
  const int seq_len = static_cast<int>(in0_data.size());

  const std::vector<int64_t> in1_data(batch_size * seq_len, 1);

  /*
   * <|im_start|>system
   * You are a helpful assistant.<|im_end|>
   * <|im_start|>user
   * 讲解一下“温故而知新”<|im_end|>
   * <|im_start|>assistant
   * “温故而知新”这句话出自《论语》，它来源于孔子的一句名言：“温故而知新，可以为师矣。”大意是说，
   * 通过温习以前学过的知识，并在新的情况下加以运用，就可以成为老师了。
   * 这句话的意思是：学习并不是孤立地接受某方面的知识，而是要把它和自己的经验联系起来。通过复习已
   * 有的知识，我们可以在新的环境中发现并理解和吸收新的东西，从而获得更多的见识和能力。
   * 具体来说，“温故”是指回顾以前所学的知识；“知新”则是指对这些旧知识有新的理解或感悟。“温故而知
   * 新”，就是在回顾已经学过的内容时，不仅要记住它们的表面意思，更要思考其背后的含义，尝试着去理
   * 解它们如何能够帮助我们解决当前的问题，或者拓展我们的视野。
   * 举个例子，如果我们正在学习一门新课程，例如编程语言，我们可以先复习一些基础知识，比如语法、数
   * 据结构等，然后尝试应用这些知识来解决问题，例如编写一个简单的程序。这样不仅加深了对这些知识的
   * 记忆，而且也能发现自己的不足之处，进而学习更多相关的知识点，从而达到“温故而知新”的效果。
   * <|im_end|>
   */
  const std::vector<int64_t> out1_data = {
      151644, 8948,   198,    2610,   525,    264,    10950,  17847,  13,
      151645, 198,    151644, 872,    198,    105250, 100158, 2073,   99416,
      99535,  68536,  52183,  16628,  854,    151645, 198,    151644, 77091,
      198,    2073,   99416,  99535,  68536,  52183,  16628,  854,    106599,
      110434, 26940,  67831,  72881,  87243,  99652,  107904, 108752, 99774,
      99700,  13072,  77144,  36987,  99416,  99535,  68536,  52183,  16628,
      3837,   73670,  17714,  99235,  107647, 32945,  26288,  36589,  20412,
      36587,  3837,   67338,  99416,  99347,  103982, 47764,  105565, 100032,
      90395,  18493,  100676, 104705, 105922, 104026, 3837,   102003, 99787,
      101049, 34187,  3407,   106599, 105855, 20412,  5122,   100134, 102095,
      118606, 29490,  100669, 99569,  104481, 100032, 3837,   103955, 30534,
      105744, 33108,  100005, 100034, 72064,  99793,  1773,   67338,  107090,
      36667,  99996,  100032, 3837,   97639,  104964, 100676, 109130, 99879,
      62926,  115167, 104460, 100676, 100413, 3837,   101982, 100350, 102075,
      112518, 33108,  99788,  3407,   100398, 99883,  41505,  99416,  99535,
      854,    104442, 106113, 103982, 31838,  47764,  107232, 24968,  2073,
      52183,  16628,  854,    104428, 63367,  32664,  100001, 100052, 100032,
      18830,  100676, 101128, 57191,  109201, 53647,  99416,  99535,  68536,
      52183,  16628,  33590,  108764, 106113, 99461,  47764,  38182,  104597,
      13343,  3837,   112589, 105712, 104017, 9370,   104386, 100313, 3837,
      109954, 104107, 41146,  109302, 109091, 3837,   104482, 99164,  85336,
      101128, 104017, 100007, 100006, 100364, 97639,  100638, 67949,  103936,
      3837,   100631, 104206, 103952, 105637, 3407,   99357,  18947,  103358,
      3837,   109198, 96555,  100134, 109747, 16628,  103995, 3837,   77557,
      110569, 102064, 3837,   105773, 60726,  107090, 101883, 115847, 3837,
      101912, 117206, 5373,   20074,  100166, 49567,  3837,   101889, 104482,
      99892,  100001, 100032, 36407,  107124, 3837,   77557,  108598, 46944,
      105172, 74220,  1773,   99654,  99902,  109566, 34187,  32664,  100001,
      100032, 109938, 3837,   101885, 104425, 99879,  100005, 102004, 105046,
      3837,   106581, 100134, 99573,  105470, 113366, 3837,   101982, 100366,
      2073,   99416,  99535,  68536,  52183,  16628,  97907,  101062, 1773,
      151645};

  std::string model_version = "3.2.0";
  test_model(model_name, batch_size, in0_data, in1_data, out1_data,
             model_version);
}
#endif
