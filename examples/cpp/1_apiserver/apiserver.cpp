/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    apiserver.cpp
 */
#ifdef EXAMPLE_MULTI_NUMA
#include <allspark/allspark_client.h>
#else
#include <allspark/allspark.h>
#endif

#include <allspark/dlpack.h>
#include <unistd.h>

#include <CLI11.hpp>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <restbed>
#include <sstream>
#include <string>
#include <thread>

#include "misc.hpp"
#include "tokenizer.h"

using namespace std;
using namespace restbed;
using namespace std::chrono;
using namespace allspark;
using json = nlohmann::json;

int default_topk = 1024;
float default_topp = 0.8;
int default_engine_max_length =
    2048;  // this length equal to input_token + max_generate_length;
int default_max_output = 512;  // max output token size.

std::string model_name = "qwen";

#ifdef EXAMPLE_MULTI_NUMA
using Engine = allspark::AsClientEngine;
#else
using Engine = allspark::AsEngine;
#endif

vector<shared_ptr<Session>> sessions;
std::unique_ptr<Engine> as_engine;
Tokenizer tokenizer;

static std::mutex g_session_mutex;

class SessionRequestContext {
 public:
  std::mutex mutex_;
  allspark::RequestHandle_t handle_;
  allspark::AsEngine::ResultQueue_t queue_;
  std::shared_ptr<AsEngine::RequestContent> req_;
  std::shared_ptr<DLTensorManager> dl_mgr_token;
  int token_count_ = 0;
  bool finished_ = false;
};

const auto headers = multimap<string, string>{
    {"Connection", "keep-alive"},
    {"Cache-Control", "no-cache"},
    {"Content-Type", "text/event-stream"},
    {"Access-Control-Allow-Origin", "*"}  // Only required for demo purposes.
};

const auto sse_headers = multimap<string, string>{
    {"Connection", "keep-alive"},
    {"Cache-Control", "no-cache"},
    {"Content-Type", "application/json"},
    {"Access-Control-Allow-Origin", "*"}  // Only required for demo purposes.
};

std::string build_block_response(const std::string& new_text) {
  // spec reference: https://huggingface.github.io/text-generation-inference/#/
  json resp = {{"generated_text", new_text}, {"details", {"seed", 12345}}};
  return resp.dump() + "\n\n";
}

std::map<std::string, std::shared_ptr<SessionRequestContext>> g_session_map;

std::shared_ptr<SessionRequestContext> GetSessionContextByUUID(
    const std::string& uuid) {
  std::unique_lock<std::mutex> locker;
  if (g_session_map.find(uuid) == std::end(g_session_map)) {
    std::cerr << "uuid not found, warn..., uuid: " << uuid << std::endl;
    return nullptr;
  }
  return g_session_map[uuid];
}

void AddNewSessionContext(const std::string& uuid,
                          allspark::RequestHandle_t handle,
                          allspark::AsEngine::ResultQueue_t queue,
                          std::shared_ptr<AsEngine::RequestContent> req,
                          std::shared_ptr<DLTensorManager> dl_mgr_token) {
  auto new_session = std::make_shared<SessionRequestContext>();
  new_session->handle_ = handle;
  new_session->queue_ = queue;
  new_session->req_ = req;
  new_session->dl_mgr_token = dl_mgr_token;

  std::unique_lock<std::mutex> locker;

  if (g_session_map.find(uuid) != std::end(g_session_map)) {
    std::cerr << "uuid already inserted in map, warn..., uuid: " << uuid
              << std::endl;
    return;
  }

  g_session_map[uuid] = new_session;
  return;
}

allspark::GenerateConfig build_qwen_generate_config(void) {
  allspark::GenerateConfig gen_config;
  gen_config.max_length = default_engine_max_length;
  gen_config.early_stopping = true;
  gen_config.eos_token_id = 151643;
  gen_config.stop_words_ids = {{151643}, {151644}, {151645}};
  gen_config.top_k = default_topk;
  gen_config.top_p = default_topp;
  gen_config.mm_info = nullptr;
  gen_config.seed = 12345;

  return std::move(gen_config);
}

void block_generate_handler(const shared_ptr<Session> session) {
  const auto request = session->get_request();

  size_t content_length = request->get_header("Content-Length", 0);

  session->fetch(content_length, [](const shared_ptr<Session>& session,
                                    const Bytes& body) {
    string body_str(reinterpret_cast<const char*>(body.data()), body.size());
    printf("body (register): size:%ld body:  %s\n", body_str.size(),
           body_str.data());

    json body_js = json::parse(body_str);

    // TODO: check inputs exists?
    auto str = body_js["inputs"].template get<std::string>();
    printf("input string: id:%s inputs: %s\n", session->get_id().c_str(),
           str.c_str());

    std::string full_prompt_text = wrap_system_prompt_qwen(str);
    // start a request, and set the generate uuid into session.

    std::vector<int64_t> tokens = tokenizer.Encode(full_prompt_text);
    auto req_ = std::make_shared<AsEngine::RequestContent>();

    auto base_gen_config = build_qwen_generate_config();
    req_->config = base_gen_config;
    req_->config.max_length = align_max_Length(
        tokens.size(), default_max_output, default_engine_max_length);

    // TODO: this token needs to store in session ?
    auto dl_mgr_token = std::make_shared<DLTensorManager>();
    dl_mgr_token->ToDLTensor({tokens});
    const DLTensorMap inputs = {{"input_ids", dl_mgr_token->GetDlTensor()}};

    req_->inputs = std::make_shared<DLTensorMap>(inputs);
    allspark::RequestHandle_t handle_;
    allspark::AsEngine::ResultQueue_t queue_;

    auto start_req_status =
        as_engine->StartRequest(model_name.c_str(), req_, &handle_, &queue_);
    if (start_req_status != allspark::AsStatus::ALLSPARK_SUCCESS) {
      std::cerr << "Start request error" << std::endl;
      return;
    }

    auto status = as_engine->SyncRequest(model_name.c_str(), handle_);
    if (status != allspark::AsStatus::ALLSPARK_SUCCESS) {
      std::cerr << "Sync request error" << std::endl;
      return;
    }

    std::shared_ptr<allspark::AsEngine::GeneratedElements> ele = nullptr;
    ele = queue_->Get();

    if (ele) {
      auto strs = tokenizer.Decode(ele->ids_from_generate);

      // get the handler uuid
      std::string uuid = generate_uuid(32);
      session->set_id(uuid);

      auto json_str = build_block_response(strs);

      session->yield(OK, json_str);

      std::ostringstream oss;
      oss << strs.size();

      session->close(OK, "Received", {{"Content-Length", oss.str()}});

      as_engine->StopRequest(model_name.c_str(), handle_);
      as_engine->ReleaseRequest(model_name.c_str(), handle_);
    } else {
      session->close(OK, "Received", {{"Content-Length", "0"}});
    }
  });
}

void stream_generate_handler(const shared_ptr<Session> session) {
  const auto request = session->get_request();

  size_t content_length = request->get_header("Content-Length", 0);

  session->fetch(content_length, [](const shared_ptr<Session>& session,
                                    const Bytes& body) {
    // 请求体现在已经被完整读取到 'body' 变量中
    string body_str(reinterpret_cast<const char*>(body.data()), body.size());
    printf("body (register): size:%ld body:  %s\n", body_str.size(),
           body_str.data());

    json body_js = json::parse(body_str);

    // TODO: check inputs exists?
    auto str = body_js["inputs"].template get<std::string>();
    printf("input string: id:%s inputs: %s\n", session->get_id().c_str(),
           str.c_str());

    std::string full_prompt_text = wrap_system_prompt_qwen(str);
    // start a request, and set the generate uuid into session.

    std::vector<int64_t> tokens = tokenizer.Encode(full_prompt_text);
    auto req_ = std::make_shared<AsEngine::RequestContent>();

    auto base_gen_config = build_qwen_generate_config();
    req_->config = base_gen_config;
    req_->config.max_length = align_max_Length(
        tokens.size(), default_max_output, default_engine_max_length);

    // TODO: this token needs to store in session ?
    auto dl_mgr_token = std::make_shared<DLTensorManager>();
    dl_mgr_token->ToDLTensor({tokens});
    const DLTensorMap inputs = {{"input_ids", dl_mgr_token->GetDlTensor()}};

    req_->inputs = std::make_shared<DLTensorMap>(inputs);
    allspark::RequestHandle_t handle_;
    allspark::AsEngine::ResultQueue_t queue_;

    auto start_req_status =
        as_engine->StartRequest(model_name.c_str(), req_, &handle_, &queue_);
    if (start_req_status != allspark::AsStatus::ALLSPARK_SUCCESS) {
      std::cerr << "Start request error" << std::endl;
      exit(1);  // TODO: more carefull handle in production code.
    }

    // get the handler uuid
    std::string uuid = generate_uuid(32);
    session->set_id(uuid);

    AddNewSessionContext(uuid, handle_, queue_, req_, dl_mgr_token);

    session->yield(OK, headers, [](const shared_ptr<Session> session) {
      sessions.push_back(session);
    });
  });
}

std::string build_stream_response(const std::string& new_text) {
  // spec reference: https://huggingface.github.io/text-generation-inference/#/
  json resp = {{"generated_text", new_text}, {"details", {"seed", 12345}}};
  return resp.dump() + "\n\n";
}

void event_stream_handler(void) {
  static size_t counter = 0;
  const auto message = "data: event " + to_string(counter) + "\n\n";

  // close session if closed.
  // TODO: release request if session is end.
  sessions.erase(std::remove_if(sessions.begin(), sessions.end(),
                                [](const shared_ptr<Session>& a) {
                                  return a->is_closed();
                                }),
                 sessions.end());

  for (auto session : sessions) {
    // get request body
    auto request = session->get_request();
    auto body = request->get_body();
    string body_str(reinterpret_cast<const char*>(body.data()), body.size());

    std::string uuid = session->get_id();
    printf("stream body: size:%ld body:  %s uuid:%s\n", body.size(),
           body.data(), session->get_id().c_str());

    auto session_ctx = GetSessionContextByUUID(uuid);
    if (!session_ctx) {
      return;
    }

    // if request already released, should not call the request handle
    if (session_ctx->finished_) continue;
    // move current latest output
    // this use no wait interface, the interface will return null if no new
    // element.
    fetch_request_output(
        as_engine, model_name, session_ctx->handle_, session_ctx->queue_,
        session_ctx->req_, tokenizer,
        [session,
         session_ctx](allspark::AsEngine::GenerateRequestStatus status) {
          std::ostringstream oss;
          oss << session_ctx->token_count_;
          session->close(OK, "Received", {{"Content-Length", oss.str()}});

          {
            std::unique_lock<std::mutex> lock(session_ctx->mutex_);
            if (!session_ctx->finished_) {
              std::cout << " [server]: send stop request.\n";
              as_engine->StopRequest(model_name.c_str(), session_ctx->handle_);

              std::cout << " [server]: send release request.\n";
              as_engine->ReleaseRequest(model_name.c_str(),
                                        session_ctx->handle_);
              session_ctx->finished_ = true;
            }
          }
        },
        [session, session_ctx](std::string new_text, int64_t new_token) {
          if (new_text.size() > 0) {
            session->yield(OK, sse_headers,
                           [](const shared_ptr<Session> session) {
                             sessions.push_back(session);
                           });
            auto json_str = build_stream_response(new_text);

            std::unique_lock<std::mutex> lock(session_ctx->mutex_);
            session_ctx->token_count_ += 1;

            std::cout << ">>   " << new_text << std::endl;

            session->yield(json_str);
          }
        });
  }

  counter++;
}

int main(int argc, const char** argv) {
  std::string model_path;
  std::string tiktoken_file;

  std::string compute_unit = "CUDA:0";
  int compute_cores = 32;
  int max_batch_size = 32;

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
                 "NUMA(by cmd lscpu)");

  app.add_option("--compute_unit", compute_unit,
                 "running device, like CUDA:0 for single GPU, CUDA:0,1 for "
                 "double GPU, CPU:10 for cpu with 10 compute threads.  ")
      ->default_str("CUDA:0");

  app.add_option("--max_batch_size", max_batch_size,
                 "max batch size for this engine, exceeds this batch size will "
                 "reject request.");

  CLI11_PARSE(app, argc, argv);

  // -----------------------
  // start setting up engine.
  // ------------------------
  auto all_exists = check_model_file_exists(model_path, tiktoken_file);
  if (!all_exists) return 1;

  std::string dimodel_file = model_path + ".dimodel";
  std::string ditensors_file = model_path + ".ditensors";

  // create an inference engine instance.
  setup_tiktoken_tokenizer(tiktoken_file, tokenizer);

  as_engine.reset(new Engine());

  // use model config builder to build model config.
  AsModelConfigBuilder model_config_builder;

  model_config_builder.withModelName(model_name)
      .withModelPath(dimodel_file)
      .withWeightsPath(ditensors_file)
      .withEngineMaxLength(default_engine_max_length)
      .withEngineMaxBatch(max_batch_size)
      .withComputeUnit(compute_unit);

  if (!begins_with(compute_unit, "CUDA")) {
    model_config_builder.withMatmulPrecision("medium").withNumThreads(
        compute_cores);  // this number is worker's thread
                         // number, use phy core(not phy-thread
                         // number) number in side of each numa
  }

  auto model_config = model_config_builder.build();

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
  // --------------------
  // setup restful server.
  // ---------------------
  auto stream_generate_resource = make_shared<Resource>();
  stream_generate_resource->set_path("/generate_stream");
  stream_generate_resource->set_method_handler("POST", stream_generate_handler);

  auto block_generate_resource = make_shared<Resource>();
  block_generate_resource->set_path("/generate");
  block_generate_resource->set_method_handler("POST", block_generate_handler);

  auto settings = make_shared<Settings>();
  settings->set_port(1984);

  auto service = make_shared<Service>();
  service->publish(stream_generate_resource);
  service->publish(block_generate_resource);
  service->schedule(event_stream_handler, chrono::milliseconds(100));
  service->start(settings);

  return EXIT_SUCCESS;
}
