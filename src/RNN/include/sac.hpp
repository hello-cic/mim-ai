#pragma once
#ifndef STRUCT_HPP
#define STRUCT_HPP

#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <unordered_map>

// 所有类和结构体

namespace _mi { // mim internal（mim的内部）
    // 工具函数
    class mi {
    public:
        static inline bool logBool;
        static inline void log(std::string content) {
            if (logBool)
                std::cout << content;
        }

        static void logOn() { logBool = true; }

        static void logOff() { logBool = false; }
    };
    /* ==========RNN（循环神经网络）的结构体即类========== */

    class rnn {
    public:
        struct rnnp {
            std::vector<float> w;               // 权重
            std::vector<float> b;              // 偏置
            std::vector<float> wa;            // 权重更改量（累加）
            std::vector<float> ba;           // 偏置更改量（累加）
            std::vector<size_t> neur;       // 每层神经元数
            std::string err = "";          // 错误信息
        };

        struct rnn_tmp {
            std::vector<float> a;      // 各层激活值（展开）
            std::vector<float> j;     // 只用隐状态跑出来的激活值
            std::vector<float> h;    // 隐状态
            float last_loss = 0.0f; // 最近一次的损失（MSE）
            size_t last_word_id = 0; // 最近一次输入的词索引（用于词向量反向传播）
            bool use_embedding = false; // 是否使用词向量
        };

        // 词向量相关
        struct embedding {
            std::unordered_map<std::string, size_t> word2id;  // 词 -> 索引
            std::vector<std::string> id2word;                  // 索引 -> 词
            std::vector<std::vector<float>> vectors;           // 词向量
            std::vector<std::vector<float>> grads;             // 词向量梯度
            size_t dim = 0;                                    // 向量维度
        };

        rnnp rp;
        rnn_tmp rt;
        embedding emb;
        float lr = 0.01f;  // 学习率

        // 初始化：neur = {输入层, 隐藏层1, 隐藏层2, ...}, h_size = 隐状态长度
        template <typename NT, size_t N>
        void init(std::array<NT, N> neur, size_t h_size = 3);

        // 词向量初始化
        void init_embedding(size_t vocab_size, size_t vec_dim);

        // 添加词到词表（返回词的索引）
        size_t add_word(std::string word);

        // 获取词向量（返回可修改的引用）
        std::vector<float>& get_vector(std::string word);

        // 前向传播：input 是当前时刻的输入，结果存到 rt.a
        void forward(std::vector<float> input);

        // 反向传播：用比较下降算法更新 wa/ba，goal = 输出层目标值
        float rd(std::vector<float> goal);

        // 一次训练：前向 + 反向，自动管理隐状态，返回损失
        float train(std::vector<float> input, std::vector<float> target);
        // 用词索引训练（自动查词向量，反向传播时更新词向量）
        float train_word(size_t word_id, std::vector<float> target);

        // 把 wa/ba 真正应用到 w/b，然后清零 wa/ba
        void apply_update();

        // 把词向量梯度应用到词向量，然后清零梯度
        void apply_embedding_update();

        // Oja Hebbian 学习：Δw = η * y * (x - y * w)，返回更新后的输出 y
        std::vector<float> hebbian(std::vector<float> x, size_t layer = 1);

        // 隐状态便捷接口
        std::vector<float> get_h();
        void set_h(std::vector<float> new_h);
    };
}

#endif