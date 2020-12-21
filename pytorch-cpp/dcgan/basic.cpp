#include <torch/torch.h>
#include <iostream>

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
  // create an identity matrix
  std::cout << "Identity Matrix" << std::endl;
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
  
  // print net parameters
  std::cout << "Print Net Parameters" << std::endl;
  Net net(4, 5);
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }
  
  // equivalent to named_parameters()
  std::cout << "Equivalent to python named_parameters()" << std::endl;
  for (const auto& pair : net.named_parameters()) {
  std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
  // running forward pass here
  std::cout << "forward pass\n" << net.forward(torch::ones({2, 4})) << std::endl;
}