#pragma once
#ifndef BPE_HPP
#define BPE_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>

namespace _mi {

    // BPE 分词器
    // 中文：先把每个字当作一个 token，然后学合并规则
    // 英文：先按字符拆分，然后学合并规则
    class bpe {
    public:
        // 合并规则：(token_a, token_b) -> merged_token
        std::vector<std::pair<std::string, std::string>> merges;
        std::unordered_map<std::string, int> vocab;  // token -> id
        int vocab_size = 0;

        // 从文本学习 BPE 合并规则
        void learn(const std::string& text, int num_merges) {
            // 1. 拆成字符序列
            std::vector<std::string> tokens;
            for (size_t i = 0; i < text.size(); ) {
                unsigned char c = text[i];
                if (c < 0x80) {
                    // ASCII 字符
                    if (c != ' ' && c != '\n' && c != '\t') {
                        tokens.push_back(std::string(1, c));
                    }
                    i++;
                } else if (c < 0xE0) {
                    // 2 字节 UTF-8
                    tokens.push_back(text.substr(i, 2));
                    i += 2;
                } else if (c < 0xF0) {
                    // 3 字节 UTF-8（中文常用）
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                } else {
                    // 4 字节 UTF-8
                    tokens.push_back(text.substr(i, 4));
                    i += 4;
                }
            }

            // 2. 学合并规则
            for (int merge_idx = 0; merge_idx < num_merges; merge_idx++) {
                // 统计相邻对出现次数
                std::unordered_map<std::string, int> pair_count;
                for (size_t i = 0; i + 1 < tokens.size(); i++) {
                    std::string pair = tokens[i] + "|||" + tokens[i + 1];
                    pair_count[pair]++;
                }

                if (pair_count.empty()) break;

                // 找最常见的对
                std::string best_pair;
                int best_count = 0;
                for (auto& [pair, count] : pair_count) {
                    if (count > best_count) {
                        best_count = count;
                        best_pair = pair;
                    }
                }

                // 分割出 a 和 b
                size_t sep = best_pair.find("|||");
                std::string a = best_pair.substr(0, sep);
                std::string b = best_pair.substr(sep + 3);

                // 记录合并规则
                merges.push_back({a, b});

                // 执行合并
                std::vector<std::string> new_tokens;
                for (size_t i = 0; i < tokens.size(); i++) {
                    if (i + 1 < tokens.size() && tokens[i] == a && tokens[i + 1] == b) {
                        new_tokens.push_back(a + b);
                        i++;  // 跳过 b
                    } else {
                        new_tokens.push_back(tokens[i]);
                    }
                }
                tokens = new_tokens;

                if (merge_idx % 100 == 0) {
                    std::cout << "merge " << merge_idx << "/" << num_merges 
                              << " (" << a << " + " << b << " -> " << a + b << ")" << std::endl;
                }
            }

            // 3. 构建词表
            build_vocab(tokens);
        }

        // 分词：先按字符拆分，再逐个应用合并规则
        std::vector<std::string> tokenize(const std::string& text) {
            // 拆成字符
            std::vector<std::string> tokens;
            for (size_t i = 0; i < text.size(); ) {
                unsigned char c = text[i];
                if (c < 0x80) {
                    if (c != ' ' && c != '\n' && c != '\t') {
                        tokens.push_back(std::string(1, c));
                    }
                    i++;
                } else if (c < 0xE0) {
                    tokens.push_back(text.substr(i, 2));
                    i += 2;
                } else if (c < 0xF0) {
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                } else {
                    tokens.push_back(text.substr(i, 4));
                    i += 4;
                }
            }

            // 逐个应用合并规则
            for (auto& [a, b] : merges) {
                std::vector<std::string> new_tokens;
                for (size_t i = 0; i < tokens.size(); i++) {
                    if (i + 1 < tokens.size() && tokens[i] == a && tokens[i + 1] == b) {
                        new_tokens.push_back(a + b);
                        i++;
                    } else {
                        new_tokens.push_back(tokens[i]);
                    }
                }
                tokens = new_tokens;
            }

            return tokens;
        }

        // token -> id
        int token_to_id(const std::string& token) {
            auto it = vocab.find(token);
            if (it != vocab.end()) return it->second;
            // 未知 token 返回 0
            return 0;
        }

        // id -> token
        std::string id_to_token(int id) {
            for (auto& [token, tid] : vocab) {
                if (tid == id) return token;
            }
            return "<unk>";
        }

        // 文本 -> id 序列
        std::vector<int> encode(const std::string& text) {
            auto tokens = tokenize(text);
            std::vector<int> ids;
            for (auto& t : tokens) {
                ids.push_back(token_to_id(t));
            }
            return ids;
        }

    private:
        // 从合并后的 tokens 构建词表
        void build_vocab(const std::vector<std::string>& tokens) {
            vocab["<unk>"] = 0;
            vocab_size = 1;

            // 统计频率
            std::unordered_map<std::string, int> freq;
            for (auto& t : tokens) {
                freq[t]++;
            }

            // 按频率排序
            std::vector<std::pair<std::string, int>> sorted(freq.begin(), freq.end());
            std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });

            for (auto& [token, count] : sorted) {
                if (vocab.find(token) == vocab.end()) {
                    vocab[token] = vocab_size++;
                }
            }
        }
    };
}

#endif
