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
//INPUTS:                   a_input, a_scale_input, a_zero_point_input, b_input, b_scale_input, b_zero_point_input, y_scale_input, y_zero_point_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class QLinearMatMul : public Layer {
        typedef struct {
            
			
            Shape_t a_input; Shape_t a_scale_input; Shape_t a_zero_point_input; Shape_t b_input; Shape_t b_scale_input; Shape_t b_zero_point_input; Shape_t y_scale_input; Shape_t y_zero_point_input;
            
            Shape_t y_output;
            
        } binding_descriptor;

        
        std::string a_input; std::string a_scale_input; std::string a_zero_point_input; std::string b_input; std::string b_scale_input; std::string b_zero_point_input; std::string y_scale_input; std::string y_zero_point_input;
        
        std::string y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QLinearMatMul(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _a_input, std::string _a_scale_input, std::string _a_zero_point_input, std::string _b_input, std::string _b_scale_input, std::string _b_zero_point_input, std::string _y_scale_input, std::string _y_zero_point_input, std::string _y_output); 

        ~QLinearMatMul() {}
    };

}

#endif

