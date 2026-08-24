#pragma once
#ifndef RUN_HPP
#define RUN_HPP

#include "sac.hpp"
#include <vector>
#include <cmath>

namespace _mi {

    // ========== 前向传播 ==========
    // 输入 input，逐层计算，把所有层的激活值存到 rt.a 里
    // rt.a 的布局：[第0层所有神经元, 第1层所有神经元, ..., 最后一层所有神经元]
    void rnn::forward(std::vector<float> input) {
        const std::vector<size_t>& tn = rp.neur;
        rt.a.clear();

        // 第0层（输入层）直接存进去
        rt.a.insert(rt.a.end(), input.begin(), input.end());

        size_t wi = 0;  // w 索引，从头开始一路加
        size_t bi = 0;  // b 索引，同理

        // 逐层往前算
        std::vector<float> old_a = input;
        for (size_t layer = 1; layer < tn.size(); layer++) {
            std::vector<float> new_a;
            for (size_t j = 0; j < tn[layer]; j++) {
                // 计算第 j 个神经元的加权和 = b + sum(w * a)
                float sum = rp.b[bi];
                for (size_t k = 0; k < tn[layer - 1]; k++) {
                    sum += rp.w[wi] * old_a[k];
                    wi++;
                }
                // 中间层用 tanh 激活，输出层不激活
                if (layer != tn.size() - 1) {
                    sum = std::tanh(sum);
                }
                new_a.push_back(sum);
                bi++;
            }
            // 把这一层的激活值也存到 rt.a 里
            rt.a.insert(rt.a.end(), new_a.begin(), new_a.end());
            old_a = new_a;
        }
    }

    // ========== 反向传播（比较下降）==========
    // a_pred = 整个网络前向传播的所有激活值
    // 用标准反向传播：从输出层误差往回传
    // 返回损失 = 输出层 err 的均方误差
    float rnn::rd() {
        const std::vector<size_t>& tn = rp.neur;
        size_t last = tn.size() - 1;

        // ---- 输出层误差 ----
        // 目标是 0，err = target - a_pred
        size_t out_start = 0;
        for (size_t i = 0; i < last; i++) {
            out_start += tn[i];
        }

        std::vector<float> err(tn[last], 0.0f);
        for (size_t j = 0; j < tn[last]; j++) {
            err[j] = 0.0f - rt.a[out_start + j];  // target=0
        }

        // ---- 标准反向传播：从输出层往回逐层更新 ----
        size_t wi = 0;
        size_t bi = 0;

        // 先计算输出层的 delta（无 tanh，delta = err）
        std::vector<float> delta = err;

        // 从最后一层往前传
        for (size_t layer = last; layer >= 1; layer--) {
            size_t prev_a_start = 0;
            for (size_t i = 0; i < layer - 1; i++) {
                prev_a_start += tn[i];
            }

            // 更新 b（delta 直接加到 ba）
            size_t b_offset = 0;
            for (size_t i = 1; i < layer; i++) {
                b_offset += tn[i];
            }
            for (size_t j = 0; j < tn[layer]; j++) {
                rp.ba[b_offset + j] += delta[j] * lr;
            }

            // 更新 w（delta * prev_a）
            size_t w_offset = 0;
            for (size_t i = 0; i < layer - 1; i++) {
                w_offset += tn[i] * tn[i + 1];
            }
            for (size_t k = 0; k < tn[layer - 1]; k++) {
                for (size_t j = 0; j < tn[layer]; j++) {
                    rp.wa[w_offset + k * tn[layer] + j] += delta[j] * rt.a[prev_a_start + k] * lr;
                }
            }

            // 计算前一层的 delta（如果有下一层）
            if (layer > 1) {
                std::vector<float> new_delta(tn[layer - 1], 0.0f);
                for (size_t k = 0; k < tn[layer - 1]; k++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < tn[layer]; j++) {
                        sum += delta[j] * rp.w[w_offset + k * tn[layer] + j];
                    }
                    // 中间层有 tanh，乘导数 (1 - a^2)
                    float a = rt.a[prev_a_start + k];
                    new_delta[k] = sum * (1.0f - a * a);
                }
                delta = new_delta;
            }
        }

        // 计算损失：MSE = 平均(err^2)
        float loss = 0.0f;
        for (size_t j = 0; j < tn[last]; j++) {
            loss += err[j] * err[j];
        }
        loss /= tn[last];
        rt.last_loss = loss;
        return loss;
    }

    // ========== 一次训练：前向 + 反向 ==========
    // 把隐状态拼到输入后面，前向传播，反向传播，返回损失
    float rnn::train(std::vector<float> input) {
        // 把隐状态拼到输入后面
        std::vector<float> full_input = input;
        full_input.insert(full_input.end(), rt.h.begin(), rt.h.end());

        // 前向传播
        forward(full_input);

        // 反向传播，更新 wa/ba，返回损失
        float loss = rd();

        // 用隐藏层激活值更新隐状态（不是输出层）
        const std::vector<size_t>& tn = rp.neur;
        size_t h_start = tn[0];  // 隐藏层在 rt.a 中的起始位置
        size_t h_count = std::min(rt.h.size(), tn[1]);  // 取较小值，防止越界
        rt.h.assign(rt.a.begin() + h_start, rt.a.begin() + h_start + h_count);

        // 隐状态防爆：软限幅，保持梯度流畅
        for (auto& val : rt.h) {
            val = std::tanh(val);
        }

        return loss;
    }

    // ========== 批量训练：依次训练多个输入，返回平均损失 ==========
    float rnn::train(std::vector<std::vector<float>> input) {
        float total_loss = 0.0f;
        for (auto& clip_input : input) {
            std::vector<float> full_input = clip_input;
            full_input.insert(full_input.end(), rt.h.begin(), rt.h.end());

            forward(full_input);
            total_loss += rd();

            // 更新隐状态（用隐藏层激活值）
            const std::vector<size_t>& tn = rp.neur;
            size_t h_start = tn[0];
            size_t h_count = std::min(rt.h.size(), tn[1]);
            rt.h.assign(rt.a.begin() + h_start, rt.a.begin() + h_start + h_count);

            // 隐状态防爆：软限幅
            for (auto& val : rt.h) {
                val = std::tanh(val);
            }
        }
        return total_loss / input.size();
    }

    // ========== 把 wa/ba 应用到 w/b ==========
    // 在一批训练结束后调用，把累加的增量应用到权重上，然后清零 wa/ba
    void rnn::apply_update() {
        for (size_t i = 0; i < rp.w.size(); i++) {
            rp.w[i] += rp.wa[i];
            rp.wa[i] = 0.0f;
        }
        for (size_t i = 0; i < rp.b.size(); i++) {
            rp.b[i] += rp.ba[i];
            rp.ba[i] = 0.0f;
        }
    }

    // ========== 隐状态便捷接口 ==========
    std::vector<float> rnn::get_h() {
        return rt.h;
    }

    void rnn::set_h(std::vector<float> new_h) {
        rt.h = new_h;
    }

}

#endif
