/*
* 本项目是RNN
* 收敛方式：比较下降：
*    正常输入要输入的和隐状态得到a_pred
*    通过 a * (w / wh)（a是自己的激活值，w是当前 分发* 神经元与当前神经元的w，wh是求和所有绝对值后的w（求和ads(w)））
*        *分发：分发的意思是从当前神经元的激活值分开成几个神经元的激活值
*    通过这个公式得到输出层是正确答案的话的激活值a_true_on_context
*    再只输入隐状态得到a_j
*    通过a_true = a_true_on_context + a_j得到a_true
*    再用err = a_true - a_pred得到误差
*    用误差求w和b增加量后，累加到wa和ba
*    最后在一次训练结束后，把wa和ba真正修改w和b，wa和ba清零
*/

/*
* 希望不要：
*    当硬编码战士
*    用deque
*    写的太抽象
*    不写注释
*    用new
*/

#define _u8

#include "RNN/RNN.hpp"
#include <iostream>

int main() {
    rnn net;
    net.init(std::array<size_t, 3>{2, 3, 1}, 2);
    net.lr = 0.0005f;
    
    // 训练 1000 次，观察 loss 变化
    for (int i = 0; i < 1000; i++) {
        float loss = net.train({0.5f, -0.3f});
        net.apply_update();
        if (i % 100 == 0) {
            std::cout << "step " << i << " loss: " << loss << std::endl;
        }
    }
    return 0;
}
