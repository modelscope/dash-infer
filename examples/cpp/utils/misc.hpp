/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    misc.hpp
 */
#pragma once

#include <allspark/dlpack.h>
#include <tokenizer.h>

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

static bool file_exists(const std::string& path) {
  std::ifstream file(path.c_str());
  return file.good();
}

static bool check_model_file_exists(const std::string& model_path,
                                    const std::string& tiktoken_file) {
  std::string asgraph_file = model_path + ".asgraph";
  std::string asparam_file = model_path + ".asparam";

  std::cout << "Model Path: " << model_path << std::endl;
  std::cout << "ASGraph File: " << asgraph_file << std::endl;
  std::cout << "ASParam File: " << asparam_file << std::endl;
  std::cout << "Token table File: " << tiktoken_file << std::endl;

  if (!file_exists(asgraph_file)) {
    std::cerr << "Error: ASGraph file does not exist at '" << asgraph_file
              << "'" << std::endl;
    return false;
  }

  if (!file_exists(asparam_file)) {
    std::cerr << "Error: ASParam file does not exist at '" << asparam_file
              << "'" << std::endl;
    return false;
  }

  if (!file_exists(tiktoken_file)) {
    std::cerr << "Error: token does not exist at '" << tiktoken_file << "'"
              << std::endl;
    return false;
  }

  return true;
}

static std::string wrap_system_prompt_qwen(const std::string& raw_text) {
  std::string prefix =
      "<|im_start|>system\nYou are a helpful "
      "assistant.<|im_end|>\n<|im_start|>user\n";

  std::string suffix = "<|im_end|>\n<|im_start|>assistant\n";

  std::string result;

  result.reserve(prefix.size() + raw_text.size() + suffix.size());
  result.append(prefix);
  result.append(raw_text);
  result.append(suffix);

  return std::move(result);
}

static void erase_previous_line() {
  // move up
  std::cout << "\033[A";
  // clear current line
  std::cout << "\033[K";
}

static void print_tokens(const std::vector<int64_t>& tokens) {
  std::cout << "std::vector<int64_t> tokens = {";
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::cout << tokens[i];
    if (i < tokens.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "}" << std::endl;
}

class DLTensorManager {
 public:
  DLTensorManager() : dl_managed_tensor_(nullptr) {}
  DLTensorManager(DLManagedTensor* dl_tensor) : dl_managed_tensor_(dl_tensor) {}
  ~DLTensorManager() {
    if (dl_managed_tensor_) {
      if (dl_managed_tensor_->deleter) {
        std::cout << "clinet destroy dl tensor" << std::endl;
        dl_managed_tensor_->deleter(dl_managed_tensor_);
        dl_managed_tensor_->deleter = nullptr;
      }
    }
  }
  void ToDLTensor(const std::vector<std::vector<int64_t>>& input) {
    if (dl_managed_tensor_ && dl_managed_tensor_->deleter) {
      dl_managed_tensor_->deleter(dl_managed_tensor_);
      dl_managed_tensor_->deleter = nullptr;
    }
    dl_managed_tensor_ = new DLManagedTensor();

    // only CPU support now
    dl_managed_tensor_->dl_tensor.device.device_id = 0;
    dl_managed_tensor_->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
    dl_managed_tensor_->dl_tensor.ndim = 2;
    dl_managed_tensor_->dl_tensor.strides = nullptr;
    int64_t* shape = new int64_t[2];
    shape[0] = input.size();
    shape[1] = input[0].size();
    // copy data
    int64_t* data = new int64_t[shape[0] * shape[1]];
    for (int i = 0; i < shape[0]; i++) {
      std::memcpy(data + i * shape[1], input[i].data(),
                  sizeof(int64_t) * shape[1]);
    }
    dl_managed_tensor_->dl_tensor.data = reinterpret_cast<void*>(data);

    dl_managed_tensor_->dl_tensor.shape = shape;
    dl_managed_tensor_->dl_tensor.dtype.lanes = 1;
    dl_managed_tensor_->dl_tensor.byte_offset = 0;
    dl_managed_tensor_->dl_tensor.dtype.code = DLDataTypeCode::kDLInt;
    dl_managed_tensor_->dl_tensor.dtype.bits = 64;
    dl_managed_tensor_->deleter = [](DLManagedTensor* self) {
      if (self) {
        if (self->dl_tensor.shape) {
          delete[] self->dl_tensor.shape;
        }
        if (self->dl_tensor.strides) {
          delete[] self->dl_tensor.strides;
        }
        if (self->dl_tensor.data) {
          delete[] static_cast<int64_t*>(self->dl_tensor.data);
        }
        delete self;
      }
    };
    dl_managed_tensor_->manager_ctx = nullptr;
  }

  void ToVectorData(std::vector<std::vector<int64_t>>& output) {
    assert(dl_managed_tensor_ && dl_managed_tensor_->dl_tensor.ndim == 2);
    // set data
    for (int i = 0; i < dl_managed_tensor_->dl_tensor.shape[0]; i++) {
      std::vector<int64_t> out(dl_managed_tensor_->dl_tensor.shape[1]);
      int data_size = dl_managed_tensor_->dl_tensor.shape[1] *
                      dl_managed_tensor_->dl_tensor.dtype.bits / 8;
      char* data_ptr =
          reinterpret_cast<char*>(dl_managed_tensor_->dl_tensor.data) +
          i * data_size;
      memcpy(out.data(), data_ptr, data_size);
      output.push_back(out);
    }
  }

  DLManagedTensor* GetDlTensor() { return dl_managed_tensor_; }

 private:
  DLManagedTensor* dl_managed_tensor_;
};

template <typename Engine, typename NewTextHandler,
          typename GenerateFinishHandler>
void fetch_request_output(
    std::unique_ptr<Engine>& as_engine, std::string model_name,
    allspark::RequestHandle_t handle_, allspark::AsEngine::ResultQueue_t queue_,
    std::shared_ptr<allspark::AsEngine::RequestContent> req_,
    allspark::Tokenizer& tokenizer,
    GenerateFinishHandler generate_finish_handler,
    NewTextHandler new_text_handler) {
  using namespace allspark;
  // start pulling on output for this request's queue
  std::shared_ptr<allspark::AsEngine::GeneratedElements> ele = nullptr;

  // this is a block queue, get will wait for output
  ele = queue_->GetNoWait();

  // if ele is null, it's either finish generation or intterrupted by some
  // case.
  if (ele == nullptr) {
    if (queue_->GenerateStatus() ==
        allspark::AsEngine::GenerateRequestStatus::GenerateFinished) {
      generate_finish_handler(queue_->GenerateStatus());
      return;
    } else if (queue_->GenerateStatus() ==
               allspark::AsEngine::GenerateRequestStatus::GenerateInterrupted) {
      // some output can be pull out.
      std::cout << "GenerateInterrupted... request id: " << std::endl;
      generate_finish_handler(queue_->GenerateStatus());
      return;
    }
  } else {
    // cal the new_text handler token by token.
    for (auto t : ele->ids_from_generate) {
      std::vector<int64_t> one_token = {t};
      std::string new_text = tokenizer.Decode(one_token);
      new_text_handler(new_text, t);
    }
  }
}

static int align_max_Length(int input_size, int output_size,
                            int max_engine_length) {
  return std::min(input_size + output_size, max_engine_length);
}

static std::string generate_uuid(const size_t len) {
  static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";

  std::stringstream ss;
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(0, sizeof(alphanum) - 2);

  for (size_t i = 0; i < len; ++i) {
    ss << alphanum[dist(engine)];
  }

  return ss.str();
}
