#ifndef CONVINTEGER_H
#define CONVINTEGER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

input: Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
input: The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. 
input: Zero point tensor for input 'x'. It's optional and default value is 0. It's a scalar, which means a per-tensor/layer quantization.
input: Scale tensor for input 'w'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M)
output: Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.
//*/
//ConvInteger
//INPUTS:                   x_input, w_input
//OPTIONAL_INPUTS:          x_zero_point_input_opt, w_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class ConvInteger : public Layer {
        typedef struct {
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            Shape_t x_input; Shape_t w_input;
            Shape_t x_zero_point_input_opt; Shape_t w_zero_point_input_opt;
            Shape_t y_output;
            
        } binding_descriptor;

        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
        std::string x_input; std::string w_input;
        std::string x_zero_point_input_opt; std::string w_zero_point_input_opt;
        std::string y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ConvInteger(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t pads, Shape_t strides);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string x_input, std::string w_input, std::string x_zero_point_input_opt, std::string w_zero_point_input_opt, std::string y_output); 

        ~ConvInteger() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ConvInteger::ConvInteger(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* ConvInteger::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ConvInteger::init() {      
    
		binding.x_input = tensor_dict[x_input]->shape();
  		binding.w_input = tensor_dict[w_input]->shape();
  		binding.x_zero_point_input_opt = tensor_dict[x_zero_point_input_opt]->shape();
  		binding.w_zero_point_input_opt = tensor_dict[w_zero_point_input_opt]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 
    }
    
    void ConvInteger::call(std::string x_input, std::string w_input, std::string x_zero_point_input_opt, std::string w_zero_point_input_opt, std::string y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/convinteger.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[x_input]->data(), *tensor_dict[w_input]->data(), *tensor_dict[x_zero_point_input_opt]->data(), *tensor_dict[w_zero_point_input_opt]->data(), *tensor_dict[y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ConvInteger, Layer>(m, "ConvInteger")
            .def(py::init<std::string, int, Shape_t, int, Shape_t, Shape_t, Shape_t> ())
            .def("forward", &ConvInteger::forward)
            .def("init", &ConvInteger::init)
            .def("call", (void (ConvInteger::*) (std::string, std::string, std::string, std::string, std::string)) &ConvInteger::call);
    }
}

#endif

/* PYTHON STUFF

*/

