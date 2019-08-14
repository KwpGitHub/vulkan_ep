#ifndef DEQUANTIZELINEAR_H
#define DEQUANTIZELINEAR_H //DequantizeLinear
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   x_input, x_scale_input
//OPTIONAL_INPUTS:          x_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct DequantizeLinear_parameter_descriptor{    
        
    };   

    struct DequantizeLinear_input_desriptor{
        Tensor* x_input; Tensor* x_scale_input;
        Tensor* x_zero_point_input_opt;
    };

    struct DequantizeLinear_output_descriptor{
        Tensor* y_output;
        
    };

    struct DequantizeLinear_binding_descriptor{
        
		
        Shape_t x_input; Shape_t x_scale_input;
        Shape_t x_zero_point_input_opt;
        Shape_t y_output;
        
    };
}


namespace backend {

    class DequantizeLinear : public Layer {
        DequantizeLinear_parameter_descriptor parameters;
        DequantizeLinear_input_desriptor      input;
        DequantizeLinear_output_descriptor    output;
        DequantizeLinear_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, DequantizeLinear_binding_descriptor>* program;
        
    public:
        DequantizeLinear(std::string, DequantizeLinear_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~DequantizeLinear() {}

    };
}

//cpp stuff
namespace backend {    
   
    DequantizeLinear::DequantizeLinear(std::string n, DequantizeLinear_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, DequantizeLinear_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dequantizelinear.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* DequantizeLinear::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<DequantizeLinear, Layer>(m, "DequantizeLinear")
            .def("forward", &DequantizeLinear::forward);    
    }*/
}

#endif
