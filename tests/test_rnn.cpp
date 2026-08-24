#include <catch2/catch_test_macros.hpp>
#include <array>
#include "sac.hpp"
#include "init.hpp"

TEST_CASE("rnn init", "[rnn]") {
    _mi::rnn net;
    std::array<size_t, 3> neur = {2, 3, 1};
    REQUIRE_NOTHROW(net.init(neur));
}
