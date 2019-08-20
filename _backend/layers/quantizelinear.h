#ifndef QUANTIZELINEAR_H
#define QUANTIZELINEAR_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.

input: N-D full precision Input tensor to be quantized.
input: Scale for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization.
input: Zero point for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization. Default value is 0 if it's not specified.
output: N-D quantized output tensor. It has same shape as input 'x'.
*/

//QuantizeLinear
//INPUTS:                   x_input, y_scale_input
//OPTIONAL_INPUTS:          y_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class QuantizeLinear : public Layer {
        typedef struct {
            
			
            Shape_t x_input; Shape_t y_scale_input;
            Shape_t y_zero_point_input_opt;
            Shape_t y_output;
            
        } binding_descriptor;

        
        std::string x_input; std::string y_scale_input;
        std::string y_zero_point_input_opt;
        std::string y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QuantizeLinear(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _x_input, std::string _y_scale_input, std::string _y_zero_point_input_opt, std::string _y_output); 

        ~QuantizeLinear() {}
    };

}

#endif

