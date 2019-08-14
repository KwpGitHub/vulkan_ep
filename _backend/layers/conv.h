#ifndef CONV_H
#define CONV_H //Conv
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, W_input
//OPTIONAL_INPUTS:          B_input_opt
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Conv_parameter_descriptor{    
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
    };   

    struct Conv_input_desriptor{
        Tensor* X_input; Tensor* W_input;
        Tensor* B_input_opt;
    };

    struct Conv_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Conv_binding_descriptor{
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        Shape_t X_input; Shape_t W_input;
        Shape_t B_input_opt;
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Conv : public Layer {
        Conv_parameter_descriptor parameters;
        Conv_input_desriptor      input;
        Conv_output_descriptor    output;
        Conv_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Conv_binding_descriptor>* program;
        
    public:
        Conv(std::string, Conv_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Conv() {}

    };
}

//cpp stuff
namespace backend {    
   
    Conv::Conv(std::string n, Conv_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Conv_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/conv.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Conv::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Conv, Layer>(m, "Conv")
            .def("forward", &Conv::forward);    
    }*/
}

#endif
