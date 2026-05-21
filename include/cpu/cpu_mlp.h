

#ifndef CPU_MLP_H
#define CPU_MLP_H

#include <vector>


class CPUMlp {
private:
    int input_size;
    int output_size;
    std::vector<float> weights;
    std::vector<float> biases;



public:
    CPUMlp(int in_size, int out_size);

    std::vector<float> forward(const std::vector<float>& input);

};

#endif 