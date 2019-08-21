#ifndef QLINEARMATMUL_H
#define QLINEARMATMUL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output.
The quantization formula is y = saturate((x / y_scale) + y_zero_point). For (x / y_scale), it is rounding to nearest ties to even.
Refer to https://en.wikipedia.org/wiki/Rounding for details. Scale and zero point must have same shape.
They must be either scalar (per tensor) or 1-D tensor (per row for 'a' and per column for 'b'). If scale and zero point are 1-D tensor,
the number of elements of scale and zero point tensor of input 'a' and output 'y' should be equal to the number of rows of input 'a',
and the number of elements of scale and zero point tensor of input 'b' should be equal to the number of columns of input 'b'.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.

input: N-dimensional quantized matrix a
input: scale of quantized input a
input: zero point of quantized input a
input: N-dimensional quantized matrix b
input: scale of quantized input b
input: zero point of quantized input b
input: scale of quantized output y
input: zero point of quantized output y
output: Quantized matrix multiply results from a * b
*/

//QLinearMatMul
//INPUTS:                   a_i, a_scale_i, a_zero_point_i, b_i, b_scale_i, b_zero_point_i, y_scale_i, y_zero_point_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace backend {   

    class QLinearMatMul : public Layer {
        typedef struct {
            
			
            Shape_t a_i; Shape_t a_scale_i; Shape_t a_zero_point_i; Shape_t b_i; Shape_t b_scale_i; Shape_t b_zero_point_i; Shape_t y_scale_i; Shape_t y_zero_point_i;
            
            Shape_t y_o;
            
        } binding_descriptor;

        
        std::string a_i; std::string a_scale_i; std::string a_zero_point_i; std::string b_i; std::string b_scale_i; std::string b_zero_point_i; std::string y_scale_i; std::string y_zero_point_i;
        
        std::string y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QLinearMatMul(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _a_i, std::string _a_scale_i, std::string _a_zero_point_i, std::string _b_i, std::string _b_scale_i, std::string _b_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearmatmul.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[a_i]->data(), *tensor_dict[a_scale_i]->data(), *tensor_dict[a_zero_point_i]->data(), *tensor_dict[b_i]->data(), *tensor_dict[b_scale_i]->data(), *tensor_dict[b_zero_point_i]->data(), *tensor_dict[y_scale_i]->data(), *tensor_dict[y_zero_point_i]->data(), *tensor_dict[y_o]->data());
        }

        ~QLinearMatMul() {}
    };
   
}
#endif

