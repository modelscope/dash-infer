/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    example_qwen.cpp
 */
#ifdef EXAMPLE_MULTI_NUMA
#include <allspark/allspark_client.h>
#else
#include <allspark/allspark.h>
#endif

#include <allspark/dlpack.h>
#include <unistd.h>

#include <CLI/CLI.hpp>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

#include "misc.hpp"
#include "tokenizer.h"

using namespace allspark;

int default_topk = 1024;
float default_topp = 0.8;
int default_engine_max_length =
    2048;  // this length equal to input_token + max_generate_length;
int default_max_output = 512;  // max output token size.

#ifdef EXAMPLE_MULTI_NUMA
using Engine = allspark::AsClientEngine;
#else
using Engine = allspark::AsEngine;
#endif

allspark::GenerateConfig build_qwen_generate_config(void);
int align_max_Length(int input_size, int output_size, int max_engine_length);
std::string start_request_and_fetch_output(
    Engine* as_engine, std::string model_name,
    std::shared_ptr<AsEngine::RequestContent> req_, int job_count,
    Tokenizer& tokenizer);

int main(int argc, char** argv) {
  std::string model_path;
  std::string tiktoken_file;
  int compute_cores = 32;

  // parse arguments.
  CLI::App app{"DashInfer AllSpark Example load and infernece a qwen model."};
  app.add_option("--modeldir,-m", model_path,
                 "dir path to converted model file ")
      ->required();
  app.add_option("--tiktoken, -t", tiktoken_file,
                 "file path to tiktoken file of model")
      ->required();
  app.add_option("--compute_cores , -c", compute_cores,
                 "compute core, suggestion setting max phy cores inside per "
                 "NUMA(by cmd lscpu)")
      ->required();

  CLI11_PARSE(app, argc, argv);

  auto all_exists = check_model_file_exists(model_path, tiktoken_file);
  if (!all_exists) return 1;

  std::string asgraph_file = model_path + ".asgraph";
  std::string asparam_file = model_path + ".asparam";

  // create an inference engine instance.
  std::unique_ptr<Engine> as_engine = std::make_unique<Engine>();

  std::string model_name = "qwen";

  Tokenizer tokenizer;

  setup_tiktoken_tokenizer(tiktoken_file, tokenizer);

  // setup build a model from a *converted* model.
  // the convert model part, please see python example,
  // it will generate a running model from
  // a huggleface type model to a allspark model,
  // file name will be like "qwen_v15.asparam, qwen_v15.asgraph"
  //

  // use model config builder to build model config.
  AsModelConfigBuilder model_config_builder;
  auto model_config =
      model_config_builder.withModelName(model_name)
          .withModelPath(asgraph_file)
          .withWeightsPath(asparam_file)
          .withEngineMaxLength(default_engine_max_length)
          .withEngineMaxBatch(16)
          .withMatmulPrecision("medium")
          .withNumThreads(compute_cores)  // this number is worker's thread
                                          // number, use phy core(not phy-thread
                                          // number) number in side of each numa
          .build();

  // build the model by engine, in this step engine will start build a
  // runable model instance in created engine.
  auto status = as_engine->BuildModelFromConfigStruct(model_config);
  if (status != allspark::AsStatus::ALLSPARK_SUCCESS) {
    std::cerr << "Error on build model, ret: " << (int)status;
    return 1;
  }

  // start this model,
  status = as_engine->StartModel(model_name.c_str());
  if (status != allspark::AsStatus::ALLSPARK_SUCCESS) {
    std::cerr << "Error on start model, ret: " << (int)status;
    return 1;
  }

  auto base_gen_config = build_qwen_generate_config();

  // dl manager to keep dlpack life cycle.
  auto dl_mgr_token = std::make_shared<DLTensorManager>();

  int in_count = 0;

  while (true) {
    // in first time, use some typical question.
    std::string raw_text = "who are you?";

    if (in_count != 0) {
      // in second time, user can input text.

      std::cout << " Input Your Prompt: ";
      std::cin >> raw_text;
    }

    // add system prompt for qwen.
    std::string full_prompt_text = wrap_system_prompt_qwen(raw_text);

    std::cout << "Input Text: " << full_prompt_text << std::endl;
    // convert text to token
    std::vector<int64_t> tokens = tokenizer.Encode(full_prompt_text);
    print_tokens(tokens);

    // prepare token into request.
    std::shared_ptr<AsEngine::RequestContent> req_;
    req_ = std::make_shared<AsEngine::RequestContent>();
    req_->config = base_gen_config;

    req_->config.max_length = align_max_Length(
        tokens.size(), default_max_output, default_engine_max_length);

    dl_mgr_token->ToDLTensor({tokens});
    const DLTensorMap inputs = {{"input_ids", dl_mgr_token->GetDlTensor()}};

    req_->inputs = std::make_shared<DLTensorMap>(inputs);

    auto output_text = start_request_and_fetch_output(
        as_engine.get(), model_name, req_, in_count, tokenizer);

    in_count++;
  }  // end while

  // stop the model before exit.
  status = as_engine->StopModel(model_name.c_str());
  if (status != allspark::AsStatus::ALLSPARK_SUCCESS) {
    std::cerr << "Error on stop model, ret: " << (int)status;
    return 1;
  }

  return 0;
}

std::string start_request_and_fetch_output(
    Engine* as_engine, std::string model_name,
    std::shared_ptr<AsEngine::RequestContent> req_, int job_count,
    Tokenizer& tokenizer) {
  RequestHandle_t handle_;  // this handle will filled by start request.
  allspark::AsEngine::ResultQueue_t queue_;

  auto start_req_status =
      as_engine->StartRequest(model_name.c_str(), req_, &handle_, &queue_);
  // set the request length by actualy length

  if (start_req_status != allspark::AsStatus::ALLSPARK_SUCCESS) {
    std::cerr << "Start request error" << std::endl;
    exit(1);
  }

  // start pulling on output for this request's queue
  std::shared_ptr<allspark::AsEngine::GeneratedElements> ele = nullptr;

  std::vector<int64_t> output_tokens;

  while (true) {
    // this is a block queue, get will wait for output
    ele = queue_->Get();

    // if ele is null, it's either finish generation or intterrupted by some
    // case.
    if (ele == nullptr) {
      if (queue_->GenerateStatus() ==
          allspark::AsEngine::GenerateRequestStatus::GenerateFinished) {
        // the generate is finished(EOS) or get max length.
        // release this request.
        as_engine->StopRequest(model_name.c_str(), handle_);
        as_engine->ReleaseRequest(model_name.c_str(), handle_);
        output_tokens.clear();
        break;
      } else if (queue_->GenerateStatus() ==
                 allspark::AsEngine::GenerateRequestStatus::
                     GenerateInterrupted) {
        // some output can be pull out.
        std::cout << "GenerateInterrupted... request id: " << std::endl;
        as_engine->StopRequest(model_name.c_str(), handle_);
        as_engine->ReleaseRequest(model_name.c_str(), handle_);
        output_tokens.clear();
        break;
      }
    }

    // there is output in the queue, print it out.
    // only generated token will in the queue.

    auto new_size = ele->ids_from_generate.size();
    auto copy_output_token = ele->ids_from_generate;

    output_tokens.insert(output_tokens.end(), copy_output_token.begin(),
                         copy_output_token.end());
    auto strs = tokenizer.Decode(output_tokens);

    std::vector<int64_t> reference_token = {
        40, 1079, 264,  3460, 4128, 1614, 3465, 553, 54364, 14817,
        13, 358,  1079, 2598, 1207, 1103, 54,   268, 13,    151645};

    auto ref_text = tokenizer.Decode(reference_token);
    // I am a large language model created by Alibaba Cloud. I am called
    // QianWen.
    std::cout << " Tokens: ";
    print_tokens(output_tokens);
    if (job_count == 0)
      std::cout << " Decode Ref.  Text: " << ref_text << std::endl;

    //      erase_previous_line();
    std::cout << " Decode Infer Text: " << strs << std::endl;
  }

  if (output_tokens.size() > 0)
    return tokenizer.Decode(output_tokens);
  else
    return "";
}

allspark::GenerateConfig build_qwen_generate_config(void) {
  allspark::GenerateConfig gen_config;
  gen_config.max_length = default_engine_max_length;
  gen_config.early_stopping = true;
  gen_config.eos_token_id = 151643;
  gen_config.stop_words_ids = {{151643}, {151644}, {151645}};
  gen_config.top_k = default_topk;
  gen_config.top_p = default_topp;
  gen_config.seed = 12345;

  return std::move(gen_config);
}
