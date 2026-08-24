#pragma once
#ifndef STRUCT_HPP
#define STRUCT_HPP

#include <vector>
#include <array>
#include <iostream>
#include <string>

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
        };

        rnnp rp;
        rnn_tmp rt;
        float lr = 0.01f;  // 学习率

        // 初始化：neur = {输入层, 隐藏层1, 隐藏层2, ...}, h_size = 隐状态长度
        template <typename NT, size_t N>
        void init(std::array<NT, N> neur, size_t h_size = 3);

        // 前向传播：input 是当前时刻的输入，结果存到 rt.a
        void forward(std::vector<float> input);

        // 反向传播：用 main.cpp 注释里的"比较下降"算法更新 wa/ba，返回损失
        float rd();

        // 一次训练：前向 + 反向，自动管理隐状态，返回损失
        float train(std::vector<float> input);
        // 批量训练：依次训练多个输入，返回总损失
        float train(std::vector<std::vector<float>> input);

        // 把 wa/ba 真正应用到 w/b，然后清零 wa/ba
        void apply_update();

        // 隐状态便捷接口
        std::vector<float> get_h();
        void set_h(std::vector<float> new_h);
    };
}

#endif