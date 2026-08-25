#include <catch2/catch_test_macros.hpp>
#include <array>
#include "sac.hpp"
#include "init.hpp"
#include "run.hpp"

TEST_CASE("rnn init", "[rnn]") {
    _mi::rnn net;
    std::array<size_t, 3> neur = {2, 3, 1};
    REQUIRE_NOTHROW(net.init(neur, 2));
    REQUIRE(net.rp.neur.size() == 3);
    REQUIRE(net.rt.h.size() == 2);
}

TEST_CASE("rnn forward", "[rnn]") {
    _mi::rnn net;
    std::array<size_t, 3> neur = {2, 3, 1};
    net.init(neur, 2);

    std::vector<float> input = {0.5f, -0.3f};
    net.forward(input);

    // forward 之后 rt.a 应该有 2+3+1=6 个值
    REQUIRE(net.rt.a.size() == 6);
}

TEST_CASE("rnn train + apply_update", "[rnn]") {
    _mi::rnn net;
    std::array<size_t, 3> neur = {2, 3, 1};
    net.init(neur, 2);

    std::vector<float> input = {0.5f, -0.3f};

    // 保存更新前的 w
    std::vector<float> w_before = net.rp.w;

    // 训练一次
    float loss = net.train(input);

    // 损失应该是一个非负数
    REQUIRE(loss >= 0.0f);
    REQUIRE(net.rt.last_loss == loss);

    // wa/ba 应该有值了（不再是全 0）
    bool has_wa = false;
    for (float v : net.rp.wa) {
        if (v != 0.0f) { has_wa = true; break; }
    }
    REQUIRE(has_wa);

    // 应用更新
    net.apply_update();

    // w 应该变了
    bool w_changed = false;
    for (size_t i = 0; i < net.rp.w.size(); i++) {
        if (net.rp.w[i] != w_before[i]) { w_changed = true; break; }
    }
    REQUIRE(w_changed);

    // wa/ba 应该被清零了
    for (float v : net.rp.wa) {
        REQUIRE(v == 0.0f);
    }
}

TEST_CASE("rnn hidden state", "[rnn]") {
    _mi::rnn net;
    std::array<size_t, 3> neur = {2, 3, 1};
    net.init(neur, 2);

    // set_h
    std::vector<float> new_h = {0.1f, 0.2f};
    net.set_h(new_h);

    // get_h
    std::vector<float> h = net.get_h();
    REQUIRE(h.size() == 2);
    REQUIRE(h[0] == 0.1f);
    REQUIRE(h[1] == 0.2f);
}

TEST_CASE("rnn multi hidden layers", "[rnn]") {
    _mi::rnn net;
    // 2输入, 3隐藏层1, 4隐藏层2, 1输出
    std::array<size_t, 4> neur = {2, 3, 4, 1};
    net.init(neur, 2);

    // 隐状态从最后一个隐藏层（neur[2]=4）取，但 h_size=2，取 min(2,4)=2
    std::vector<float> input = {0.5f, -0.3f};
    float loss = net.train(input);

    REQUIRE(loss >= 0.0f);
    REQUIRE(net.rt.h.size() == 2);

    // 再训练一次，隐状态应该更新了
    float loss2 = net.train(input);
    REQUIRE(loss2 >= 0.0f);
}
