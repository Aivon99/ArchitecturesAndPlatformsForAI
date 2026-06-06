#include "common/tensor.h"

CpuTensor::CpuTensor(size_t h = 1, size_t w = 1, size_t d = 1) : Tensor(h, w, d) {
    data,resize(size(), 0.0f);
}

float* CpuTensor::get_raw(){
    return data.data();
}

float& CpuTensor::at(size_t h, size_t w, size_t d){
    if(h >= height || w>= width || d >= depth ){
        throw std::out_of_range("selected Position out of range");
    }
    index = flatten_index(d, h, w)
    return data[index]
}

const float& CpuTensor::at(size_t h, size_t w, size_t d){
    if(h >= height || w>= width || d >= depth ){
        throw std::out_of_range("selected Position out of range");
    }
    
    index = flatten_index(d, h, w)
    return data[index]
}
