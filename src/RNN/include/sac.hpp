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
            std::vector<float> h;
        };

        rnnp rp;
        rnn_tmp rt;

        template <typename NT, size_t N>
        void init(std::array<NT, N> neur, size_t h = 3);

        void run(std::vector<float> input);

        inline void ctbv(std::vector<float> h); // Calculate the basic value（计算基础值）

        void rd(); // Relatively decreased（比较下降）
    };
}

#endif