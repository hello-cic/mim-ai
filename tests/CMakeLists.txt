add_executable(tests test_rnn.cpp)
target_link_libraries(tests PRIVATE RNN)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# 手动注册测试，不使用 catch_discover_tests（避免 Xcode 编译时执行测试被 kill）
add_test(NAME RNNTests COMMAND tests)
