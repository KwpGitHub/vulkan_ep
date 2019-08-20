#ifndef QLINEARCONV_H
#define QLINEARCONV_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.

input: Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
input: Scale tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.
input: Zero point tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.
input: The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. 
input: Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).
input: Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).
input: Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.
input: Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.
input: Optional 1D bias to be added to the convolution, has size of M.
output: Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.
*/

//QLinearConv
//INPUTS:                   x_i, x_scale_i, x_zero_point_i, w_i, w_scale_i, w_zero_point_i, y_scale_i, y_zero_point_i
//OPTIONAL_INPUTS:          B_i
//OUTPUS:                   y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t

//class stuff
namespace backend {   

    class QLinearConv : public Layer {
        typedef struct {
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            Shape_t x_i; Shape_t x_scale_i; Shape_t x_zero_point_i; Shape_t w_i; Shape_t w_scale_i; Shape_t w_zero_point_i; Shape_t y_scale_i; Shape_t y_zero_point_i;
            Shape_t B_i;
            Shape_t y_o;
            
        } binding_descriptor;

        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
        std::string x_i; std::string x_scale_i; std::string x_zero_point_i; std::string w_i; std::string w_scale_i; std::string w_zero_point_i; std::string y_scale_i; std::string y_zero_point_i;
        std::string B_i;
        std::string y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QLinearConv(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _auto_pad,  Shape_t _dilations,  int _group,  Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides); 
        void bind(std::string _x_i, std::string _x_scale_i, std::string _x_zero_point_i, std::string _w_i, std::string _w_scale_i, std::string _w_zero_point_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _B_i, std::string _y_o); 

        ~QLinearConv() {}
    };

}

#endif

