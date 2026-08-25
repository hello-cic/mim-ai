/*
* 本项目是RNN
*
* 架构：
*   BPE 分词 → Oja Hebbian 词向量 → 比较下降 RNN
*
* 语料：维基百科中文
* 目标：用 Oja 学词向量，用比较下降做下一个词预测
*/

#include "RNN/RNN.hpp"
#include "RNN/include/bpe.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

// 读取语料文件前 N 行
std::string read_corpus(const std::string& path, size_t max_lines) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << path << std::endl;
        return "";
    }

    std::string text;
    std::string line;
    size_t count = 0;
    while (std::getline(file, line) && count < max_lines) {
        // 跳过空行和标题行
        if (!line.empty() && line[0] != '=' && line[0] != 'C') {
            text += line + " ";
            count++;
        }
    }
    std::cout << "读取了 " << count << " 行语料" << std::endl;
    return text;
}

// 归一化向量
void normalize(std::vector<float>& v) {
    float max_val = 0.0f;
    for (float x : v) max_val = std::max(max_val, std::abs(x));
    if (max_val > 1e-6f) {
        for (float& x : v) x /= max_val;
    }
}

int main() {
    // ========== 1. 读取语料 ==========
    std::cout << "=== 读取语料 ===" << std::endl;
    // 优先用项目里的本地副本，没有就用外置硬盘的完整语料
    std::string corpus = read_corpus("src/data/zhwiki_50k.txt", 50000);
    if (corpus.empty()) {
        corpus = read_corpus(
            "/Volumes/大城编程固态硬盘/语料/wiki/zhwiki_s.txt", 
            50000
        );
    }
    if (corpus.empty()) {
        std::cerr << "语料为空，退出" << std::endl;
        return 1;
    }

    // ========== 2. 学习 BPE ==========
    std::cout << "\n=== 学习 BPE ===" << std::endl;
    // 只用前 20000 个字符学 BPE，加快速度
    std::string bpe_corpus = corpus.substr(0, 20000);
    _mi::bpe tokenizer;
    tokenizer.learn(bpe_corpus, 100);  // 学 100 个合并规则
    std::cout << "词表大小: " << tokenizer.vocab_size << std::endl;

    // ========== 3. 构建词向量表 ==========
    // 把 BPE 词表转成 embedding 的 word2id
    size_t vocab_size = tokenizer.vocab_size;
    size_t vec_dim = 8;   // 词向量维度
    size_t hidden_dim = 16; // 隐藏层维度

    // 第一个网络：Oja Hebbian（无监督）
    _mi::rnn oja;
    oja.init(std::array<size_t, 2>{vec_dim, hidden_dim}, 0);
    oja.lr = 0.001f;

    // 初始化词向量
    oja.init_embedding(vocab_size, vec_dim);

    // 用随机值初始化词向量
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (size_t i = 0; i < vocab_size; i++) {
        auto& vec = oja.emb.vectors[i];
        for (size_t j = 0; j < vec_dim; j++) {
            vec[j] = dist(rng);
        }
    }

    // 第二个网络：比较下降 RNN（有监督）
    // 输出层 = 词表大小，用来预测下一个 token 的概率
    _mi::rnn supervised;
    supervised.init(std::array<size_t, 3>{hidden_dim, hidden_dim, vocab_size}, 2);
    supervised.lr = 0.0001f;

    // ========== 4. 训练 ==========
    std::cout << "\n=== 训练 ===" << std::endl;

    // 把语料转成 id 序列
    std::vector<int> token_ids = tokenizer.encode(corpus);
    std::cout << "语料 token 数: " << token_ids.size() << std::endl;

    // 训练参数
    int epochs = 2;
    int batch_size = 32;
    int seq_len = 10;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int step = 0;

        // 随机打乱位置
        std::vector<size_t> positions(token_ids.size());
        for (size_t i = 0; i < positions.size(); i++) positions[i] = i;
        std::shuffle(positions.begin(), positions.end(), rng);

        for (size_t i = 0; i + seq_len < token_ids.size() && i < 500; i += seq_len) {
            // 取一个序列
            std::vector<int> seq(token_ids.begin() + i, token_ids.begin() + i + seq_len);

            // Oja 提取特征
            std::vector<std::vector<float>> features;
            for (int id : seq) {
                std::vector<float> feat = oja.hebbian(oja.emb.vectors[id]);
                normalize(feat);
                features.push_back(feat);
            }

            // 比较下降 RNN 训练（预测下一个 token）
            supervised.set_h(std::vector<float>(2, 0.0f));
            for (size_t j = 0; j + 1 < features.size(); j++) {
                supervised.train(features[j]);
            }
            supervised.apply_update();

            total_loss += supervised.rt.last_loss;
            step++;

            if (step % 100 == 0) {
                std::cout << "epoch " << epoch << " step " << step 
                          << " loss: " << total_loss / step << std::endl;
            }
        }

        std::cout << "epoch " << epoch << " 完成，平均 loss: " << total_loss / step << std::endl;
    }

    // ========== 5. 测试 ==========
    std::cout << "\n=== 测试 ===" << std::endl;

    // 测试几个词的相似度
    auto test_similarity = [&](const std::string& w1, const std::string& w2) {
        auto ids1 = tokenizer.encode(w1);
        auto ids2 = tokenizer.encode(w2);
        if (ids1.empty() || ids2.empty()) {
            std::cout << w1 << " vs " << w2 << ": 未知词" << std::endl;
            return;
        }
        auto& v1 = oja.emb.vectors[ids1[0]];
        auto& v2 = oja.emb.vectors[ids2[0]];

        float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
        for (size_t i = 0; i < vec_dim; i++) {
            dot += v1[i] * v2[i];
            n1 += v1[i] * v1[i];
            n2 += v2[i] * v2[i];
        }
        float sim = dot / (std::sqrt(n1) * std::sqrt(n2) + 1e-8f);
        std::cout << w1 << " vs " << w2 << ": " << sim << std::endl;
    };

    test_similarity("数学", "物理");
    test_similarity("数学", "苹果");
    test_similarity("中国", "美国");

    // 预测下一个词
    std::cout << "\n=== 预测下一个词 ===" << std::endl;
    std::string test_text = "数学是";
    auto test_ids = tokenizer.encode(test_text);
    std::cout << "输入: " << test_text << " -> ids: ";
    for (int id : test_ids) std::cout << id << " ";
    std::cout << std::endl;

    supervised.set_h(std::vector<float>(2, 0.0f));
    for (int id : test_ids) {
        std::vector<float> feat = oja.hebbian(oja.emb.vectors[id]);
        normalize(feat);
        supervised.forward(feat);
    }

    // 输出层做 softmax，找概率最高的 token
    size_t out_start = 0;
    for (size_t i = 0; i < supervised.rp.neur.size() - 1; i++) {
        out_start += supervised.rp.neur[i];
    }
    size_t out_size = supervised.rp.neur.back();

    // softmax
    std::vector<float> probs(out_size);
    float max_val = *std::max_element(
        supervised.rt.a.begin() + out_start,
        supervised.rt.a.begin() + out_start + out_size
    );
    float sum = 0.0f;
    for (size_t i = 0; i < out_size; i++) {
        probs[i] = std::exp(supervised.rt.a[out_start + i] - max_val);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    // 找 top 5
    std::vector<std::pair<float, int>> top;
    for (size_t i = 0; i < out_size; i++) {
        top.push_back({probs[i], i});
    }
    std::sort(top.begin(), top.end(), std::greater<>());

    std::cout << "Top 5 预测:" << std::endl;
    for (int k = 0; k < 5 && k < (int)top.size(); k++) {
        std::string token = tokenizer.id_to_token(top[k].second);
        std::cout << "  " << token << " (" << top[k].first << ")" << std::endl;
    }

    return 0;
}
