#ifndef CONVTRANSPOSE_H
#define CONVTRANSPOSE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    
input: Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn)
input: The weight tensor that will be used in the convolutions; has size (C x M/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is the dimension of the kernel. The number of channels in the output should be equal to W.shape[1] * group (assuming zero based indices of the shape array)
input: Optional 1D bias to be added to the convolution, has size of M.
output: Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, pad lengths and group count. The number of channels in the output should be equal to W.shape[1] * group (assuming zero based indices of the shape array)
//*/
//ConvTranspose
//INPUTS:                   X_input, W_input
//OPTIONAL_INPUTS:          B_input_opt
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class ConvTranspose : public Layer {
        typedef struct {
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t output_padding; Shape_t output_shape; Shape_t pads; Shape_t strides;
			
            Shape_t X_input; Shape_t W_input;
            Shape_t B_input_opt;
            Shape_t Y_output;
            
        } binding_descriptor;

        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t output_padding; Shape_t output_shape; Shape_t pads; Shape_t strides;
        std::string X_input; std::string W_input;
        std::string B_input_opt;
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ConvTranspose(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t output_padding, Shape_t output_shape, Shape_t pads, Shape_t strides);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string W_input, std::string B_input_opt, std::string Y_output); 

        ~ConvTranspose() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ConvTranspose::ConvTranspose(std::string n, int auto_pad, Shape_t dilations, int group, Shape_t kernel_shape, Shape_t output_padding, Shape_t output_shape, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* ConvTranspose::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ConvTranspose::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.W_input = tensor_dict[W_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.output_padding = output_padding;
  		binding.output_shape = output_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 
    }
    
    void ConvTranspose::call(std::string X_input, std::string W_input, std::string B_input_opt, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/convtranspose.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[W_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ConvTranspose, Layer>(m, "ConvTranspose")
            .def(py::init<std::string, int, Shape_t, int, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t> ())
            .def("forward", &ConvTranspose::forward)
            .def("init", &ConvTranspose::init)
            .def("call", (void (ConvTranspose::*) (std::string, std::string, std::string, std::string)) &ConvTranspose::call);
    }
}

#endif

/* PYTHON STUFF

*/

