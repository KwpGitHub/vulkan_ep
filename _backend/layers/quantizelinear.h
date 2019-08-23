#ifndef QUANTIZELINEAR_H
#define QUANTIZELINEAR_H 

#include "../layer.h"

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
//INPUTS:                   x_i, y_scale_i
//OPTIONAL_INPUTS:          y_zero_point_i
//OUTPUS:                   y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class QuantizeLinear : public backend::Layer {
        typedef struct {          
            backend::Shape_t x_i; backend::Shape_t y_scale_i;
            backend::Shape_t y_zero_point_i;
            backend::Shape_t y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        std::string x_i; std::string y_scale_i;
        std::string y_zero_point_i;
        std::string y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        QuantizeLinear(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _x_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o); 
        virtual void build();

        ~QuantizeLinear() {}
    };
   
}
#endif

