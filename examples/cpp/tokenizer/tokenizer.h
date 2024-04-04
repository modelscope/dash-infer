/**
 * This code is a C++ tokenizer for qwen model,
 * provided by Banma network (https://www.ebanma.com/),
 * and modified based on fastllm.cpp
 * (https://github.com/ztxz16/fastllm/blob/bf0be781e2106b1429f1be7cf829b9d7c26bf75d/src/fastllm.cpp).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace allspark {

struct Tokenizer {
  enum TokenizerType { BPE = 0, NORMAL = 1, QWEN = 2 };

  struct TrieNode {
    int tokenId;
    float score;
    std::map<int, TrieNode*> next;
    TrieNode();
  };
  struct Symbol {
    TrieNode* node;
    char* s;
    int pos, len;
    int prev, next;
    int fixId;

    Symbol(Tokenizer::TrieNode* node, char* s, int pos, int len, int prev,
           int next, int fixId) {
      this->node = node;
      this->s = s;
      this->pos = pos;
      this->len = len;
      this->prev = prev;
      this->next = next;
      this->fixId = fixId;
    }
  };
  struct SymbolPairs {
    float score;
    int l, r, size;

    SymbolPairs(float score, int l, int r, int size) {
      this->score = score;
      this->l = l;
      this->r = r;
      this->size = size;
    }
  };

  friend bool operator<(const SymbolPairs& a, const SymbolPairs& b) {
    return a.score < b.score || (a.score == b.score && a.l > b.l);
  }

  TrieNode* root;

  TokenizerType type = TokenizerType::BPE;

  std::unordered_map<int, std::string> tokenToStringDict;
  std::unordered_map<int, float> tokenToScoreDict;
  std::unordered_map<std::string, uint64_t> stringToTokenDict;

  std::mutex mutex_;

  Tokenizer();

  ~Tokenizer();

  void Clear();  // clear

  void TryMergePairs(
      std::vector<Symbol>& symbols, int l, int r,
      std::priority_queue<SymbolPairs>& q);  // insert backup symbol

  // insert one token to table
  void Insert(const std::string& s, int64_t tokenId, float score = 1.0f);

  std::vector<int64_t> Encode(const std::string& s);

  std::string Decode(const std::vector<int64_t>& tokens);
};

void setup_tiktoken_tokenizer(std::string tiktoken_path,
                              allspark::Tokenizer& tokenizer);
}  // namespace allspark
