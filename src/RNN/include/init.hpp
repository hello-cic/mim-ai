#pragma once
#ifndef INIT_HPP
#define INIT_HPP

#include "sac.hpp"
#include <vector>
#include <random>

namespace _mi {
    template <typename NT, size_t N>
    inline void rnn::init(std::array<NT, N> neur, size_t h_size) {
        // 随机数引擎
        std::random_device rd;
        std::mt19937 gen(rd());

        // Xavier 初始化：权重范围 [-1/sqrt(fan_in), 1/sqrt(fan_in)]
        // fan_in = 上一层神经元数，这样每层输出的方差比较一致
        std::normal_distribution<float> w_dist(0.0f, 1.0f);

        // 输入层存原始输入维度，不加 h_size
        // h_size 会在 train() 里拼接到输入后面
        rp.neur.assign(neur.begin(), neur.end());
        mi::log("neur初始化成功");

        // 初始化 w：每层用 Xavier 初始化
        // 第一层的 fan_in = 原始输入 + 隐状态（因为 train 会拼接）
        for (size_t i = 1; i < rp.neur.size(); i++) {
            float fan_in;
            if (i == 1) {
                fan_in = static_cast<float>(rp.neur[0] + h_size);  // 第一层要考虑 h
            } else {
                fan_in = static_cast<float>(rp.neur[i - 1]);
            }
            float std_dev = 1.0f / std::sqrt(fan_in);

            for (size_t j = 0; j < rp.neur[i] * (i == 1 ? rp.neur[0] + h_size : rp.neur[i - 1]); j++) {
                rp.w.push_back(w_dist(gen) * std_dev);
            }
        }

        // 初始化 b：全 0
        for (size_t i = 1; i < rp.neur.size(); i++)
            rp.b.resize(rp.b.size() + rp.neur[i], 0.0f);

        // 初始化增量数组
        rp.wa.resize(rp.w.size(), 0.0f);
        rp.ba.resize(rp.b.size(), 0.0f);

        // 初始化临时变量
        // rt.a 要存所有层的激活值，第一层实际大小是 neur[0] + h_size
        rt.a.resize(rp.b.size() + h_size, 0.0f);
        rt.j.resize(rt.a.size(), 0.0f);

        // 隐状态初始化为 0
        rt.h.resize(h_size, 0.0f);
    }
}

#endif
