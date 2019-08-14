#ifndef CONVTRANSPOSE_H
#define CONVTRANSPOSE_H //ConvTranspose
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, W_input
//OPTIONAL_INPUTS:          B_input_opt
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ConvTranspose_parameter_descriptor{    
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t output_padding; Shape_t output_shape; Shape_t pads; Shape_t strides;
    };   

    struct ConvTranspose_input_desriptor{
        Tensor* X_input; Tensor* W_input;
        Tensor* B_input_opt;
    };

    struct ConvTranspose_output_descriptor{
        Tensor* Y_output;
        
    };

    struct ConvTranspose_binding_descriptor{
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t output_padding; Shape_t output_shape; Shape_t pads; Shape_t strides;
		
        Shape_t X_input; Shape_t W_input;
        Shape_t B_input_opt;
        Shape_t Y_output;
        
    };
}


namespace backend {

    class ConvTranspose : public Layer {
        ConvTranspose_parameter_descriptor parameters;
        ConvTranspose_input_desriptor      input;
        ConvTranspose_output_descriptor    output;
        ConvTranspose_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ConvTranspose_binding_descriptor>* program;
        
    public:
        ConvTranspose(std::string, ConvTranspose_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ConvTranspose() {}

    };
}

//cpp stuff
namespace backend {    
   
    ConvTranspose::ConvTranspose(std::string n, ConvTranspose_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ConvTranspose_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/convtranspose.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ConvTranspose::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ConvTranspose, Layer>(m, "ConvTranspose")
            .def("forward", &ConvTranspose::forward);    
    }*/
}

#endif
