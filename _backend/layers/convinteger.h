#ifndef CONVINTEGER_H
#define CONVINTEGER_H //ConvInteger
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   x_input, w_input
//OPTIONAL_INPUTS:          x_zero_point_input_opt, w_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ConvInteger_parameter_descriptor{    
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
    };   

    struct ConvInteger_input_desriptor{
        Tensor* x_input; Tensor* w_input;
        Tensor* x_zero_point_input_opt; Tensor* w_zero_point_input_opt;
    };

    struct ConvInteger_output_descriptor{
        Tensor* y_output;
        
    };

    struct ConvInteger_binding_descriptor{
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        Shape_t x_input; Shape_t w_input;
        Shape_t x_zero_point_input_opt; Shape_t w_zero_point_input_opt;
        Shape_t y_output;
        
    };
}


namespace backend {

    class ConvInteger : public Layer {
        ConvInteger_parameter_descriptor parameters;
        ConvInteger_input_desriptor      input;
        ConvInteger_output_descriptor    output;
        ConvInteger_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ConvInteger_binding_descriptor>* program;
        
    public:
        ConvInteger(std::string, ConvInteger_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ConvInteger() {}

    };
}

//cpp stuff
namespace backend {    
   
    ConvInteger::ConvInteger(std::string n, ConvInteger_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ConvInteger_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/convinteger.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ConvInteger::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ConvInteger, Layer>(m, "ConvInteger")
            .def("forward", &ConvInteger::forward);    
    }*/
}

#endif
