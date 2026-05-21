#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>


class Tensor {
protected:
    size_t height;
    size_t width;
    size_t depth;
    
public:
    virtual ~Tensor() = default;

    Tensor(int h = 1, int w = 1, int d = 1)
        : height(h), width(w), depth(d)
    {
        if (h <= 0 || w <= 0 || d <= 0) { //i assume a vector, 1D, of length N is N,1, 1
            throw std::invalid_argument("tensor dimensions must be positive");
        }
    }
 
    size_t get_height() const { return height; }
    size_t get_width() const { return width; }
    size_t get_depth() const { return depth; }

    size_t size() const {
        return height * width * depth;
    }
    virtual float* get_raw() = 0;
    virtual float& at(size_t h, size_t w, size_t d) = 0;
    virtual const float& at(size_t d, size_t h, size_t w) const = 0;

};


class CpuTensor : public Tensor {
private:
    std::vector<float> data; // 1D vector to hold tensor data in row-major order

    size_t flatten_index(size_t d, size_t h = 0, size_t w = 0) const {
        return d* ( width * depth) + 
                (h * width) +
                 w;
    }

public:



    CpuTensor(size_t h = 1,size_t w = 1,size_t d = 1) : Tensor(h, w, d) {
        data.resize(size(), 0.0f); // Initialize with zeros
    }


    float* get_raw() override {
        return data.data();
    }

 
    float& at(size_t h, size_t w,size_t d) override{
        if (h < 0 || h >= height || w < 0 || w >= width || d < 0 || d >= depth) {
            throw std::out_of_range("tensor index out of range");
        
        }
        
        return data[flatten_index(d,h,w)];

    }

    const float& at(size_t h, size_t w, size_t d) const override{
        if (h < 0 || h >= height || w < 0 || w >= width || d < 0 || d >= depth) {
            throw std::out_of_range("tensor index out of range");
        
        }
        
        return data[flatten_index(d,h,w)];

    }




};


class GpuTensor : public Tensor {

private:


public:


};




#endif

