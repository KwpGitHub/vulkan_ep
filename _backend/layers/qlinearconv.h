#ifndef QLINEARCONV_H
#define QLINEARCONV_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
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
//INPUTS:                   x_input, x_scale_input, x_zero_point_input, w_input, w_scale_input, w_zero_point_input, y_scale_input, y_zero_point_input
//OPTIONAL_INPUTS:          B_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class QLinearConv : public Layer {
        typedef struct {    
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
        } parameter_descriptor;  

        typedef struct {
            Tensor* x_input; Tensor* x_scale_input; Tensor* x_zero_point_input; Tensor* w_input; Tensor* w_scale_input; Tensor* w_zero_point_input; Tensor* y_scale_input; Tensor* y_zero_point_input;
            Tensor* B_input_opt;
        } input_desriptor;

        typedef struct {
            Tensor* y_output;
            
        } output_descriptor;

        typedef struct {
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
            Shape_t x_input; Shape_t x_scale_input; Shape_t x_zero_point_input; Shape_t w_input; Shape_t w_scale_input; Shape_t w_zero_point_input; Shape_t y_scale_input; Shape_t y_zero_point_input;
            Shape_t B_input_opt;
            Shape_t y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QLinearConv(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~QLinearConv() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    QLinearConv::QLinearConv(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearconv.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* QLinearConv::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void QLinearConv::init() {
		binding.x_input = input.x_input->shape();
  		binding.x_scale_input = input.x_scale_input->shape();
  		binding.x_zero_point_input = input.x_zero_point_input->shape();
  		binding.w_input = input.w_input->shape();
  		binding.w_scale_input = input.w_scale_input->shape();
  		binding.w_zero_point_input = input.w_zero_point_input->shape();
  		binding.y_scale_input = input.y_scale_input->shape();
  		binding.y_zero_point_input = input.y_zero_point_input->shape();
  		binding.B_input_opt = input.B_input_opt->shape();
 
		binding.y_output = output.y_output->shape();
 
		binding.auto_pad = parameters.auto_pad;
  		binding.dilations = parameters.dilations;
  		binding.group = parameters.group;
  		binding.kernel_shape = parameters.kernel_shape;
  		binding.pads = parameters.pads;
  		binding.strides = parameters.strides;
 
        program->bind(binding, *input.x_input->data(), *input.x_scale_input->data(), *input.x_zero_point_input->data(), *input.w_input->data(), *input.w_scale_input->data(), *input.w_zero_point_input->data(), *input.y_scale_input->data(), *input.y_zero_point_input->data(), *input.B_input_opt->data(), *output.y_output->data());
    }
    
    void QLinearConv::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<QLinearConv, Layer>(m, "QLinearConv")
            .def("forward", &QLinearConv::forward);    
    }
}*/

#endif
