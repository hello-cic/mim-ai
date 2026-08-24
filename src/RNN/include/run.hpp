#pragma once
#ifndef RUN_HPP
#define RUN_HPP

#include "sac.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

namespace _mi {

    void rnn::run(std::vector<float> input) {
        std::vector<size_t> tn = rp.neur;

        // 初始化索引
        size_t wi = 0;
        size_t bi = 0;

        // 初始化前面的值
        std::vector<float> old_a = input;
        std::vector<float> new_a;

        rt.a = input;

        for (size_t i = 1; i < tn.size(); i++) {
            new_a.resize(0);
            for (size_t j = 0; j < tn[i]; j++) {
                float my_a = 0.0f;
                for (size_t y = 0; y < tn[i - 1]; y++) {
                    my_a += rp.w[wi] * old_a[y];
                    wi++;
                }
                my_a += rp.b[bi];
                if (i != tn.size() - 1)
                    my_a = std::tanh(my_a);

                new_a.push_back(my_a);
                rt.a.push_back(my_a);
                bi++;
            }
            old_a = new_a;
        }
    }

    void rnn::ctbv(std::vector<float> h) {
        size_t n = rp.neur[0] - h.size();  // 要加的0个数
        
        // input 头部插入 n 个 0
        std::vector<float> input;
        input.insert(input.begin(), n, 0.0f);

        // 真-inline：
        {
            std::vector<size_t> tn = rp.neur;

            // 初始化索引
            size_t wi = 0;
            size_t bi = 0;

            // 初始化前面的值
            std::vector<float> old_a = input;
            std::vector<float> new_a;

            rt.a = input;

            for (size_t i = 1; i < tn.size(); i++) {
                new_a.resize(0);
                for (size_t j = 0; j < tn[i]; j++) {
                    float my_a = 0.0f;
                    for (size_t y = 0; y < tn[i - 1]; y++) {
                        my_a += rp.w[wi] * old_a[y];
                        wi++;
                    }
                    my_a += rp.b[bi];
                    if (i != tn.size() - 1)
                        my_a = std::tanh(my_a);

                    new_a.push_back(my_a);
                    rt.j.push_back(my_a);
                    bi++;
                }
                old_a = new_a;
            }
        }
    }

    void rnn::rd() {
        
    }
}

#endif