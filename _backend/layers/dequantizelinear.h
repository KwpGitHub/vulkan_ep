#ifndef DEQUANTIZELINEAR_H
#define DEQUANTIZELINEAR_H 

#include "../layer.h"

/*

The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).

input: N-D quantized input tensor to be de-quantized.
input: Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization.
input: Zero point for input 'x'. It's a scalar, which means a per-tensor/layer quantization. It's optional. 0 is the default value when it's not specified.
output: N-D full precision output tensor. It has same shape as input 'x'.
*/

//DequantizeLinear
//INPUTS:                   x_i, x_scale_i
//OPTIONAL_INPUTS:          x_zero_point_i
//OUTPUS:                   y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class DequantizeLinear : public backend::Layer {
        typedef struct {          
            backend::Shape_t x_i; backend::Shape_t x_scale_i;
            backend::Shape_t x_zero_point_i;
            backend::Shape_t y_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        
        std::string x_i; std::string x_scale_i;
        std::string x_zero_point_i;
        std::string y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        DequantizeLinear(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _x_i, std::string _x_scale_i, std::string _x_zero_point_i, std::string _y_o); 
        virtual void build();

        ~DequantizeLinear() {}
    };
   
}
#endif

