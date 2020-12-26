/*
Credit to original authors who develop this code in pytorch c++ tutorial
reference: https://github.com/pytorch/examples/tree/master/cpp/dcgan
*/

#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream>

void showIdentityMatrix() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
};
// a simple neural network
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

int main() {
  showIdentityMatrix();
  // define a simple network inputs
  Net net(4, 5);
  // display values of network parameters
  for (const auto& p : net.parameters()) {
      std::cout << p << std::endl;
  }
  // display names of network parameters
  for (const auto& pair : net.named_parameters()) {
  std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
  //run a fowward propagation
  std::cout << net.forward(torch::ones({2, 4})) << std::endl;
}