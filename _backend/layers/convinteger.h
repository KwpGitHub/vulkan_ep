#ifndef CONVINTEGER_H
#define CONVINTEGER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

input: Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
input: The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. 
input: Zero point tensor for input 'x'. It's optional and default value is 0. It's a scalar, which means a per-tensor/layer quantization.
input: Scale tensor for input 'w'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M)
output: Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.
*/

//ConvInteger
//INPUTS:                   x_i, w_i
//OPTIONAL_INPUTS:          x_zero_point_i, w_zero_point_i
//OUTPUS:                   y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t


//class stuff
namespace backend {   

    class ConvInteger : public Layer {
        typedef struct {
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            Shape_t x_i; Shape_t w_i;
            Shape_t x_zero_point_i; Shape_t w_zero_point_i;
            Shape_t y_o;
            
        } binding_descriptor;

        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
        std::string x_i; std::string w_i;
        std::string x_zero_point_i; std::string w_zero_point_i;
        std::string y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ConvInteger(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _auto_pad,  Shape_t _dilations,  int _group,  Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides); 
        virtual void bind(std::string _x_i, std::string _w_i, std::string _x_zero_point_i, std::string _w_zero_point_i, std::string _y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/convinteger.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[x_i]->data(), *tensor_dict[w_i]->data(), *tensor_dict[x_zero_point_i]->data(), *tensor_dict[w_zero_point_i]->data(), *tensor_dict[y_o]->data());
        }

        ~ConvInteger() {}
    };
   
}
#endif

