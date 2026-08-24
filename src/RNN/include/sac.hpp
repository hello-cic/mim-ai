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
        bool logBool = false;
        inline void log(std::string content) {
            if (logBool)
                std::cout << content;
        }

        void logOn() { logBool = true; }

        void logOff() { logBool = false; }
    };
    /* ==========RNN（循环神经网络）的结构体即类========== */

    class rnn {
        struct rnnp {
            std::vector<float> w;           // 权重
            std::vector<float> b;          // 偏置
            std::vector<float> wa;        // 权重更改量
            std::vector<float> ba;       // 偏置更改量
            std::vector<size_t> neur;   // 神经元数
            std::string err = "";      // 错误
        };

        struct rnn_tmp {
            std::vector<float> a;  // 激活值
            std::vector<float> j; // 基础值
        };

        rnnp rp;
        rnn_tmp rt;

        template <typename NT, size_t N>
        void init(std::array<NT, N> neur);

        template <typename NT, size_t N>
        void run(std::array<NT, N> input, std::array<NT, N> neur = nullptr);
    };
}

#endif