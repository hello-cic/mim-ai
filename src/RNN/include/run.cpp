#pragma once
#ifndef RUN_HPP
#define RUN_HPP

#include "sac.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace _mi {
    template <typename NT, size_t N>
    void rnn::run(std::array<NT, N> input, std::array<NT, N> neur = nullptr) {
        std::vector<float> tn;
        if (!neur = nullptr)
            tn.assign(neur.begin(), neur.end());
        else
            tn = rp.neur;
        
        // 初始化索引
        size_t wi = 0;
        size_t bi = 0;

        // 初始化前面的值
        std::vector<float> old_a = input;
        std::vector<float> new_a;

        for (size_t i = 1; i < rp.neur.size(); i++) {
            new_a.resize(0);
            for (size_t j = 0; j < rp.neur[i]; j++) {
                float my_a = 0.0f;
                for (size_t y = 0; y < rp.neur[i - 1]; y++) {
                    my_a += rp.w[wi] * old_a[y];
                    wi++;
                }
                my_a += rp.b[bi];
                if (!i == rp.neur.size() - 1)
                    my_a = std::tanh(my_a);
                
                new_a.push_back(my_a);
            }
            old_a = new_a;
        }
    }
};

#endif