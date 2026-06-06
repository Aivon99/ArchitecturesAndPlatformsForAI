

#ifndef CPU_MLP_H
#define CPU_MLP_H

#include <vector>
#include <common/tensor.h>


class CPUMlp {
private:
    
    // kind of a stud for now 
    size_t n_layers; 
    size_t input_size;
    size_t output_size;

    // each layer has a) a vector of weights (flattened matrix) and b) a vector of biases
    // pass pointers to those objects 

    std::vector<float>* tensors ;
    std::vector<float>*  biases ;

public:
    CPUMlp(size_t n_layers, size_t in_size, size_t out_size){
        this->n_layers = n_layers;
        this->input_size = in_size;
        this->output_size = out_size;
        tensors = new std::vector<float>[n_layers];
        biases = new std::vector<float>[n_layers];

        for (size_t i = 0; i < n_layers; ++i) {
            size_t layer_input_size = (i == 0) ? input_size : output_size;
            size_t layer_output_size = output_size;
            tensors[i].resize(layer_input_size * layer_output_size, 0.01f); // dummy weight init
            biases[i].resize(layer_output_size, 0.0f); // dummy bias init
        }
    };

    CPUMlp( vector<float>* tensors, vector<float>* biases) {
        this->tensors = tensors;
        this->biases = biases;
    }


    std::vector<float> forward(const std::vector<float>& input);





};

#endif 