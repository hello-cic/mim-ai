#pragma once
#ifndef INIT_HPP
#define INIT_HPP

#include "sac.hpp"
#include <vector>
#include <random>
#include <algorithm>

namespace _mi {
    template <typename NT, size_t N>
    inline void rnn::init(std::array<NT, N> neur) {
        // 随机数引擎
        std::random_device rd;
        std::mt19937 gen(rd());

        // 概率控制：20% 概率跳到边界 ±0.5（或 0.05）
        const double JUMP_PROB = 0.2;
        std::uniform_real_distribution<> prob(0.0, 1.0);

        // 核心区 [-20, 20] 对应 [-0.20, 0.20]
        std::uniform_int_distribution<> center_dis(-20, 20);

        // 边界符号选择
        std::uniform_int_distribution<> edge_sign(0, 1);


        rp.neur.assign(neur.begin(), neur.end());
        mi::log("neur初始化成功");
        

        // 初始化w
        for (size_t i = 1; i < rp.neur.size(); i++) {
            for (size_t j = 0; j < rp.neur[i] * rp.neur[i - 1]; j++) {
                int candidate_int;
                double rnd = prob(gen);

                if (rnd < (1.0 - JUMP_PROB)) {
                    candidate_int = center_dis(gen); // 例：-15, 3, 20
                } else {
                    // 跳转时只生成 -50 或 50（即 -0.5f 或 0.5f）
                    int sign = edge_sign(gen);
                    candidate_int = (sign == 0) ? -50 : 50;
                }

                // 除以 100.0f 得到 float
                float candidate = candidate_int / 100.0f;

                // 添加
                rp.neur.push_back(candidate);
            }
        }


        // 初始化b
        for (size_t i = 1; i < rp.neur.size(); i++)
            rp.b.resize(rp.b.size() + rp.neur[i], 0.0f);


        // 零零散散的其他初始化
        rp.wa.resize(rp.w.size());       // 初始化w的增加量（数量是w的数量）

        rp.ba.resize(rp.b.size());     // 初始化b的增加量（数量是b的数量)

        rt.a.resize(rp.b.size());    // 初始化a（数量是b的数量)

        rt.j.resize(rp.a.size());  // 初始化基础值j（数量是a的数量）
    }
}

#endif