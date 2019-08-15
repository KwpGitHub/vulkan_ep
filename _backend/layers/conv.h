#ifndef CONV_H
#define CONV_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

The convolution operator consumes an input tensor and a filter, and
computes the output.
input: Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
input: The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. 
input: Optional 1D bias to be added to the convolution, has size of M.
output: Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.
//*/
//Conv
//INPUTS:                   X_input, W_input
//OPTIONAL_INPUTS:          B_input_opt
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class Conv : public Layer {
        typedef struct {
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            Shape_t X_input; Shape_t W_input;
            Shape_t B_input_opt;
            Shape_t Y_output;
            
        } binding_descriptor;

        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
        std::string X_input; std::string W_input;
        std::string B_input_opt;
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Conv(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t pads, Shape_t strides);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string W_input, std::string B_input_opt, std::string Y_output); 

        ~Conv() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Conv::Conv(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* Conv::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Conv::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.W_input = tensor_dict[W_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 
    }
    
    void Conv::call(std::string X_input, std::string W_input, std::string B_input_opt, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/conv.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[W_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Conv, Layer>(m, "Conv")
            .def(py::init<std::string, int, Shape_t, int, Shape_t, Shape_t, Shape_t> ())
            .def("forward", &Conv::forward)
            .def("init", &Conv::init)
            .def("call", (void (Conv::*) (std::string, std::string, std::string, std::string)) &Conv::call);
    }
}

#endif

/* PYTHON STUFF

*/

