/**
 * This code is a C++ tokenizer for qwen model,
 * provided by Banma network (https://www.ebanma.com/),
 * and modified based on fastllm.cpp
 * (https://github.com/ztxz16/fastllm/blob/bf0be781e2106b1429f1be7cf829b9d7c26bf75d/src/fastllm.cpp).
 */

#include "tokenizer.h"

#include "base64.h"

namespace allspark {

Tokenizer::TrieNode::TrieNode() { this->tokenId = -999999; }

Tokenizer::Tokenizer() { root = new TrieNode(); }

Tokenizer::~Tokenizer() {
  Clear();
  delete root;
}

void Tokenizer::Clear() {
  std::unique_lock<std::mutex> lock(mutex_);

  std::vector<TrieNode*> q;
  q.push_back(root);
  for (int i = 0; i < q.size(); i++) {
    TrieNode* now = q[i];
    for (auto it : now->next) {
      q.push_back(it.second);
    }
  }
  root = new TrieNode();
  tokenToStringDict.clear();
}

void Tokenizer::Insert(const std::string& s, int64_t tokenId, float score) {
  std::unique_lock<std::mutex> lock(mutex_);

  TrieNode* now = this->root;
  for (int i = 0; i < s.size(); i++) {
    if (now->next.find(s[i]) == now->next.end()) {
      now->next[s[i]] = new TrieNode();
    }
    now = now->next[s[i]];
  }
  now->tokenId = tokenId;
  now->score = score;
  tokenToStringDict[tokenId] = s;
  stringToTokenDict[s] = tokenId;
}

void Tokenizer::TryMergePairs(std::vector<Symbol>& symbols, int l, int r,
                              std::priority_queue<SymbolPairs>& q) {
  if (l == -1 || r == -1 || symbols[l].len == 0 || symbols[r].len == 0) {
    return;
  }
  auto now = symbols[l].node;
  char* s = symbols[r].s;
  int pos = symbols[r].pos, len = symbols[r].len;
  for (int i = pos; i < pos + len; i++) {
    if (now->next.find(s[i]) != now->next.end()) {
      now = now->next[s[i]];
    } else {
      return;
    }
  }
  if (now->tokenId == -999999) {
    return;
  }
  q.push(SymbolPairs(151645 - now->tokenId, l, r,
                     symbols[l].len + symbols[r].len));
}

std::vector<int64_t> Tokenizer::Encode(const std::string& ori) {
  std::unique_lock<std::mutex> lock(mutex_);

  std::map<std::string, int> specialTokens = {{"<|im_start|>", 151644},
                                              {"<|im_end|>", 151645},
                                              {"<|endoftext|>", 151643}};

  // comment these special tokens for now
  // for (int i = 0; i < 205; i++) {
  //     specialTokens.insert("<|extra_" + std::to_string(i) + "|>");
  // }

  std::vector<std::pair<int, int>> sep;
  for (auto& token : specialTokens) {
    int pos = 0;
    while ((pos = ori.find(token.first, pos)) != std::string::npos) {
      sep.push_back({pos, token.first.size()});
      pos += token.first.size();
    }
  }
  sep.push_back({ori.size(), 1});  // use this to tokenize the last few words
  std::sort(sep.begin(), sep.end(), std::greater<std::pair<int, int>>());

  std::vector<Symbol> symbols;
  std::vector<int64_t> v;
  for (int i = 0; i <= ori.size(); i++) {
    if (i == sep.back().first) {
      if (!symbols.empty()) {
        symbols.back().next = -1;
        std::priority_queue<SymbolPairs> workQueue;
        for (int j = 1; j < symbols.size(); j++) {
          TryMergePairs(symbols, j - 1, j, workQueue);
        }

        while (!workQueue.empty()) {
          auto top = workQueue.top();
          workQueue.pop();
          if (symbols[top.l].len == 0 || symbols[top.r].len == 0 ||
              symbols[top.l].len + symbols[top.r].len != top.size) {
            continue;
          }

          for (int j = symbols[top.r].pos;
               j < symbols[top.r].pos + symbols[top.r].len; j++) {
            symbols[top.l].node =
                symbols[top.l].node->next[symbols[top.r].s[j]];
          }
          symbols[top.l].len += symbols[top.r].len;
          symbols[top.r].len = 0;
          symbols[top.l].next = symbols[top.r].next;
          if (symbols[top.r].next >= 0) {
            symbols[symbols[top.r].next].prev = top.l;
          }

          TryMergePairs(symbols, symbols[top.l].prev, top.l, workQueue);
          TryMergePairs(symbols, top.l, symbols[top.l].next, workQueue);
        }

        for (int j = 0; j < symbols.size(); j++) {
          if (symbols[j].len > 0) {
            v.push_back(symbols[j].node->tokenId);
          } else if (symbols[j].node == nullptr) {
            // 未识别的字符
            uint8_t c = (uint8_t)(symbols[j].s[symbols[j].pos]);
            std::string now = "<0x00>";
            now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
            now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
            if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
              v.push_back(stringToTokenDict[now]);
            }
          }
        }
        symbols.clear();
      }

      std::string special = ori.substr(sep.back().first, sep.back().second);
      if (specialTokens.find(special) != specialTokens.end()) {
        v.push_back(specialTokens[special]);
      }

      i += sep.back().second - 1;
      sep.pop_back();

      continue;
    }

    int tokenId = -999999;
    int pos = i - 1;
    TrieNode* now = this->root;
    for (int j = i; j < ori.size(); j++) {
      if (now->next.find(ori[j]) != now->next.end()) {
        now = now->next[ori[j]];
        if (now->tokenId != -999999) {
          tokenId = now->tokenId;
          pos = j;
          break;
        }
      } else {
        break;
      }
    }

    if (pos >= i) {
      symbols.push_back(Symbol(now, (char*)ori.data(), i, pos - i + 1,
                               (int)symbols.size() - 1, (int)symbols.size() + 1,
                               -999999));
      i = pos;
    } else {
      symbols.push_back(Symbol(nullptr, (char*)ori.data(), i, 0,
                               (int)symbols.size() - 1, (int)symbols.size() + 1,
                               -999999));
    }
  }

  return v;
}

std::string Tokenizer::Decode(const std::vector<int64_t>& tokens) {
  std::unique_lock<std::mutex> lock(mutex_);

  std::string ret = "";

  int needBytes = 0;

  for (int i = 0; i < tokens.size(); i++) {
    if (tokens[i] == -1) break;

    std::string s = tokenToStringDict[tokens[i]];
    if (s.size() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
      int c = 0;
      for (int i = 3; i < 5; i++) {
        c *= 16;
        if (s[i] >= '0' && s[i] <= '9') {
          c += (s[i] - '0');
        } else {
          c += (s[i] - 'A' + 10);
        }
      }

      s = " ";
      s[0] = c;
    }
    if (s == "<n>") {
      ret += "\n";
    } else if (s == "<|tab|>") {
      ret += "\t";
    } else {
      if (needBytes >= s.size()) {
        needBytes -= s.size();
      } else {
        int k = needBytes;
        needBytes = 0;
        for (; k < s.size(); k++) {
          char ch = s[k];
          if (needBytes == 0) {
            if ((ch & 0x80) == 0x00) {
              // The first 128 characters (US-ASCII) in UTF-8 format only need
              // one byte.
              continue;
            } else if ((ch & 0xE0) == 0xC0) {
              // The next 1,920 characters need two bytes to encode,
              // which covers the remainder of almost all Latin-script
              // alphabets.
              needBytes = 1;
            } else if ((ch & 0xF0) == 0xE0) {
              // Three bytes are needed for characters in the rest of
              // the Basic Multilingual Plane, which contains virtually all
              // characters in common use, including most Chinese, Japanese and
              // Korean characters.
              needBytes = 2;
            } else if ((ch & 0xF8) == 0xF0) {
              // Four bytes are needed for characters in the other planes of
              // Unicode, which include less common CJK characters, various
              // historic scripts, mathematical symbols, and emoji (pictographic
              // symbols).
              needBytes = 3;
            }
            // int leftBytes = s.size()-k-1;
            // if(leftBytes > needBytes) {
            //     needBytes = 0;
            //     k += needBytes;
            // }
            // else {
            //     needBytes = needBytes - leftBytes;
            //     break;
            // }
          } else {
            needBytes--;
          }
        }
      }

      ret += s;
    }
  }

  if (needBytes > 0) return "";

  std::string blank = "";
  blank += 226, blank += 150, blank += 129;
  while (true) {
    std::string::size_type pos(0);
    if ((pos = ret.find(blank)) != std::string::npos)
      ret.replace(pos, blank.length(), " ");
    else
      break;
  }
  int pos = ret.find("<|blank_");
  if (pos != -1) {
    int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
    return std::string(space_num, ' ');
  }

  return ret;
}

void setup_tiktoken_tokenizer(std::string tiktoken_path,
                              allspark::Tokenizer& tokenizer) {
  std::ifstream s(tiktoken_path);
  std::string line;
  while (getline(s, line)) {
    size_t pos = line.find(" ");
    if (pos != std::string::npos) {
      std::string strx = line.substr(0, pos);
      std::string strid = line.substr(pos + 1);
      std::string x = base64_decode(strx);
      uint64_t id = atoi(strid.c_str());
      tokenizer.Insert(x, id);
    }
  }
}

}  // namespace allspark
