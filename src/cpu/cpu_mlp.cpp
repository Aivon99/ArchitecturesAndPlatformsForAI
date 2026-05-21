#include "cpu/cpu_mlp.h" 
#include <cmath>

CPUMlp::CPUMlp(int in_size, int out_size) : input_size(in_size), output_size(out_size) {
    weights.resize(input_size * output_size, 0.01f); // dummy weight init
    biases.resize(output_size, 0.0f);               // dummy bias init
}
 
// Forward pass definition: performs a simple matrix-vector dot product (Y = XW + b)

std::vector<float> CPUMlp::forward(const std::vector<float>& input) {
    std::vector<float> output(output_size, 0.0f);

    for (int o = 0; o < output_size; ++o) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[o * input_size + i];
        }
        output[o] = sum + biases[o]; // Apply bias
    }

    return output;


}








