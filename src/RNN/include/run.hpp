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
                // 所有层都用 tanh 激活
                sum = std::tanh(sum);
                new_a.push_back(sum);
                bi++;
            }
            // 把这一层的激活值也存到 rt.a 里
            rt.a.insert(rt.a.end(), new_a.begin(), new_a.end());
            old_a = new_a;
        }
    }

    // ========== 反向传播（比较下降）==========
    // 算法：
    //   1. forward(input + h) → a_pred（整个网络激活值）
    //   2. 复制 rt.a，把输出层设为 goal，然后用 a * (w/wh) 从输出层往回逐层算 a_true_on_context
    //   3. forward(只有 h) → a_j（整个网络激活值）
    //   4. a_true = a_true_on_context + a_j（逐元素）
    //   5. err = a_true - a_pred（逐元素）
    //   6. 标准反向传播更新 w/b
    float rnn::rd(std::vector<float> goal) {
        const std::vector<size_t>& tn = rp.neur;
        size_t last = tn.size() - 1;

        // ---- a_pred = 当前整个网络的激活值 ----
        std::vector<float> a_pred = rt.a;

        // ---- a_true_on_context：复制 a_pred，输出层设为 goal ----
        std::vector<float> a_true_on_context = a_pred;
        size_t out_start = 0;
        for (size_t i = 0; i < last; i++) {
            out_start += tn[i];
        }
        for (size_t j = 0; j < tn[last]; j++) {
            a_true_on_context[out_start + j] = goal[j];
        }

        // ---- a_true = a_true_on_context（直接用，不加 a_j）----
        std::vector<float> a_true = a_true_on_context;

        // ---- err = a_true - a_pred（逐元素）----
        std::vector<float> err(a_pred.size(), 0.0f);
        for (size_t i = 0; i < a_pred.size(); i++) {
            err[i] = a_true[i] - a_pred[i];
        }

        // ---- 标准反向传播：从输出层往回逐层更新 ----
        std::vector<float> delta = err;

        for (size_t layer = last; layer >= 1; layer--) {
            size_t prev_a_start = 0;
            for (size_t i = 0; i < layer - 1; i++) {
                prev_a_start += tn[i];
            }

            // 更新 b
            size_t b_offset = 0;
            for (size_t i = 1; i < layer; i++) {
                b_offset += tn[i];
            }
            for (size_t j = 0; j < tn[layer]; j++) {
                rp.ba[b_offset + j] += delta[j] * lr;
            }

            // 更新 w
            size_t w_offset = 0;
            for (size_t i = 0; i < layer - 1; i++) {
                w_offset += tn[i] * tn[i + 1];
            }
            for (size_t k = 0; k < tn[layer - 1]; k++) {
                for (size_t j = 0; j < tn[layer]; j++) {
                    rp.wa[w_offset + k * tn[layer] + j] += delta[j] * rt.a[prev_a_start + k] * lr;
                }
            }

            // 计算前一层的 delta
            if (layer > 1) {
                std::vector<float> new_delta(tn[layer - 1], 0.0f);
                for (size_t k = 0; k < tn[layer - 1]; k++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < tn[layer]; j++) {
                        sum += delta[j] * rp.w[w_offset + k * tn[layer] + j];
                    }
                    float a = rt.a[prev_a_start + k];
                    new_delta[k] = sum * (1.0f - a * a);
                }
                delta = new_delta;
            }
        }

        // 如果使用词向量，把输入层的梯度传给词向量
        if (rt.use_embedding) {
            size_t copy_count = std::min(emb.dim, delta.size());
            for (size_t i = 0; i < copy_count; i++) {
                emb.grads[rt.last_word_id][i] += delta[i];
            }
        }

        // 计算损失：MSE（只看输出层）
        float loss = 0.0f;
        for (size_t j = 0; j < tn[last]; j++) {
            loss += err[out_start + j] * err[out_start + j];
        }
        loss /= tn[last];
        rt.last_loss = loss;
        return loss;
    }

    // ========== 一次训练：前向 + 反向 ==========
    // 把隐状态拼到输入后面，前向传播，反向传播，返回损失
    float rnn::train(std::vector<float> input, std::vector<float> target) {
        // 把隐状态拼到输入后面
        std::vector<float> full_input = input;
        full_input.insert(full_input.end(), rt.h.begin(), rt.h.end());

        // 前向传播
        forward(full_input);

        // 反向传播，传入目标，更新 wa/ba，返回损失
        float loss = rd(target);

        // 用最后一个隐藏层的激活值更新隐状态
        const std::vector<size_t>& tn = rp.neur;
        // 最后一个隐藏层的起始位置 = 输入层 + 所有中间隐藏层
        size_t h_start = tn[0];
        for (size_t i = 1; i < tn.size() - 2; i++) {
            h_start += tn[i];
        }
        size_t h_count = tn[tn.size() - 2];  // 倒数第二层 = 最后一个隐藏层
        size_t copy_count = std::min(rt.h.size(), h_count);
        rt.h.assign(rt.a.begin() + h_start, rt.a.begin() + h_start + copy_count);

        // 隐状态防爆：软限幅，保持梯度流畅
        for (auto& val : rt.h) {
            val = std::tanh(val);
        }

        return loss;
    }

    // ========== 用词索引训练 ==========
    float rnn::train_word(size_t word_id, std::vector<float> target) {
        rt.last_word_id = word_id;
        rt.use_embedding = true;
        float loss = train(emb.vectors[word_id], target);
        rt.use_embedding = false;
        return loss;
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

    // ========== 把词向量梯度应用到词向量 ==========
    void rnn::apply_embedding_update() {
        for (size_t i = 0; i < emb.vectors.size(); i++) {
            for (size_t j = 0; j < emb.dim; j++) {
                emb.vectors[i][j] += emb.grads[i][j] * lr;
                emb.grads[i][j] = 0.0f;
            }
        }
    }

    // ========== Oja Hebbian 学习 ==========
    // Δw = η * y * (x - y * w)
    // x: 输入向量（词向量）
    // layer: 用哪一层的权重（默认 1 = 第一个隐藏层）
    // 返回: 输出 y = x @ w
    std::vector<float> rnn::hebbian(std::vector<float> x, size_t layer) {
        const std::vector<size_t>& tn = rp.neur;

        // 计算 w 的起始位置
        size_t w_offset = 0;
        for (size_t i = 0; i < layer - 1; i++) {
            w_offset += tn[i] * tn[i + 1];
        }

        size_t in = tn[layer - 1];   // 输入维度
        size_t out = tn[layer];       // 输出维度

        // y = x @ w（前向传播）
        std::vector<float> y(out, 0.0f);
        for (size_t j = 0; j < out; j++) {
            for (size_t k = 0; k < in; k++) {
                y[j] += x[k] * rp.w[w_offset + k * out + j];
            }
        }

        // Δw = η * y * (x - y * w)
        for (size_t k = 0; k < in; k++) {
            for (size_t j = 0; j < out; j++) {
                float yw = 0.0f;
                for (size_t m = 0; m < out; m++) {
                    yw += y[m] * rp.w[w_offset + k * out + m];
                }
                // 简化：只用 y[j] 而不是全量 yw
                float delta_w = lr * y[j] * (x[k] - y[j] * rp.w[w_offset + k * out + j]);
                rp.w[w_offset + k * out + j] += delta_w;
            }
        }

        return y;
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
