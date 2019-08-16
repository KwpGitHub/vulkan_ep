#include "../layer.h"
#ifndef DEQUANTIZELINEAR_H
#define DEQUANTIZELINEAR_H 
/*

The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).

input: N-D quantized input tensor to be de-quantized.
input: Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization.
input: Zero point for input 'x'. It's a scalar, which means a per-tensor/layer quantization. It's optional. 0 is the default value when it's not specified.
output: N-D full precision output tensor. It has same shape as input 'x'.
//*/
//DequantizeLinear
//INPUTS:                   x_input, x_scale_input
//OPTIONAL_INPUTS:          x_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class DequantizeLinear : public Layer {
        typedef struct {
            
			
            Shape_t x_input; Shape_t x_scale_input;
            Shape_t x_zero_point_input_opt;
            Shape_t y_output;
            
        } binding_descriptor;

        
        std::string x_input; std::string x_scale_input;
        std::string x_zero_point_input_opt;
        std::string y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        DequantizeLinear(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string x_input, std::string x_scale_input, std::string x_zero_point_input_opt, std::string y_output); 

        ~DequantizeLinear() {}

    };
    
}

#endif

