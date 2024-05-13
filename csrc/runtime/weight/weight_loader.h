/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_loader.h
 */
#pragma once
#include <common/common.h>
#include <core/tensor/tensor.h>

#include "allspark.pb.h"
namespace allspark {

struct TensorInfo {
  Shape shape;
  DataType dtype;
  DataMode mode;
  SplitMode split_mode;
  int nnz;
  std::vector<int> group_list;
};

class WeightFileParser {
 public:
  WeightFileParser();

  // parse the tensor info from current stream's
  TensorInfo ParseTensorInfo(FILE* fp);

  TensorInfo ParseTensorInfo(const void* ptr, size_t len);

  size_t TensorHeaderBytes();
};

/**
 * Single weight loader, which inlcude the partiation for multiple rank.
 */
class WeightLoader {
 public:
  // add memory unload(swap) weight storage part, store the unsegmented
  // weight, but also don't store the buffer again
  WeightLoader(TensorInfo& info_p, RankInfo& rank_info_p,
               const std::string& name_p)
      : tensor_info_(info_p), rank_info_(rank_info_p), name_(name_p) {}

  /** load one weight from file stream
   *
   * @param fp  file pointer
   * @param output tensor.
   * */
  virtual void LoadFromFileStream(FILE* fp,
                                  std::shared_ptr<AsTensor> out_tensor) = 0;

  /** Load one weight from memory.
   *
   * @param ptr  start addr of memory wait to load.
   * @param len  input memory length wait to load.
   * @param opt_in_tensor  [optional] input tensor(whole weight, can be null
   * if don't have any), it's an optimize for no split mode.
   * @param out_tensor                output tensor.
   */
  virtual void LoadFromMemory(const void* ptr, size_t len,
                              std::shared_ptr<AsTensor> opt_in_tensor,
                              std::shared_ptr<AsTensor> out_tensor) = 0;

  virtual ~WeightLoader(){};

 protected:
  TensorInfo tensor_info_;
  RankInfo rank_info_;
  const std::string name_;
};

class SparseWeightLoader : public WeightLoader {
 public:
  SparseWeightLoader(TensorInfo& info_p, RankInfo& r_info,
                     const std::string& name_p)
      : WeightLoader(info_p, r_info, name_p) {}

  virtual void LoadFromFileStream(FILE* fp,
                                  std::shared_ptr<AsTensor> out_tensor);

  virtual void LoadFromMemory(const void* ptr, size_t len,
                              std::shared_ptr<AsTensor> opt_in_tensor,
                              std::shared_ptr<AsTensor> out_tensor);
};

class DenseWeightLoader : public WeightLoader {
 public:
  DenseWeightLoader(TensorInfo& info_p, RankInfo& r_info,
                    const std::string& name_p,
                    TensorMap* model_weight_save_buffer)
      : WeightLoader(info_p, r_info, name_p) {}

  virtual void LoadFromFileStream(FILE* fp,
                                  std::shared_ptr<AsTensor> out_tensor);

  virtual void LoadFromMemory(const void* ptr, size_t len,
                              std::shared_ptr<AsTensor> opt_in_tensor,
                              std::shared_ptr<AsTensor> out_tensor);

 private:
  std::shared_ptr<AsTensor> whole_weight_tensor_;
};

class WeightSplitter {
 public:
  WeightSplitter(SplitMode mode, RankInfo rank_info)
      : mode_(mode), rank_info_(rank_info) {}

  /**
   * Function judge a weight tensor can split or not.
   *
   * @param tensor_info
   *
   * @return  return ture if this tensor and rank info can split.
   */
  virtual bool IsSplittable(TensorInfo& tensor_info) = 0;

  /**
   * check if this mode can process, for visitor mode
   *
   * @param mode
   *
   * @return return true if can process.
   */
  virtual bool IsModelProcessable(SplitMode mode) = 0;

  /**
   * @param dst_tensor_info
   * @param dst_tensor
   * @param opt_whole_tensor  whole tensor pointer, optional, if not provided,
   * the weight pointer must be provided.
   * @param whole_weight_ptr ptr of whole weight
   * @param len the length of whole weight
   */
  virtual void CopyWeight(TensorInfo dst_tensor_info,
                          std::shared_ptr<AsTensor> dst_tensor,
                          std::shared_ptr<AsTensor> opt_whole_tensor,
                          const void* whole_weight_ptr, size_t len) = 0;
  virtual void SetShape(TensorInfo dst_tensor_info,
                        std::shared_ptr<AsTensor> dst_tensor) = 0;

  virtual ~WeightSplitter() {}

 protected:
  SplitMode mode_;
  RankInfo rank_info_;
};

/**
 * Factory method of Weight Splitter
 */

class WeightSplitterFactory {
 public:
  static std::unique_ptr<WeightSplitter> GetSplitterByMode(SplitMode mode,
                                                           RankInfo rankInfo);
};

}  // namespace allspark
