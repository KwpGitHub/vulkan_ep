#ifndef QUANTIZELINEAR_H
#define QUANTIZELINEAR_H //QuantizeLinear
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   x_input, y_scale_input
//OPTIONAL_INPUTS:          y_zero_point_input_opt
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct QuantizeLinear_parameter_descriptor{    
        
    };   

    struct QuantizeLinear_input_desriptor{
        Tensor* x_input; Tensor* y_scale_input;
        Tensor* y_zero_point_input_opt;
    };

    struct QuantizeLinear_output_descriptor{
        Tensor* y_output;
        
    };

    struct QuantizeLinear_binding_descriptor{
        
		
        Shape_t x_input; Shape_t y_scale_input;
        Shape_t y_zero_point_input_opt;
        Shape_t y_output;
        
    };
}


namespace backend {

    class QuantizeLinear : public Layer {
        QuantizeLinear_parameter_descriptor parameters;
        QuantizeLinear_input_desriptor      input;
        QuantizeLinear_output_descriptor    output;
        QuantizeLinear_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, QuantizeLinear_binding_descriptor>* program;
        
    public:
        QuantizeLinear(std::string, QuantizeLinear_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~QuantizeLinear() {}

    };
}

//cpp stuff
namespace backend {    
   
    QuantizeLinear::QuantizeLinear(std::string n, QuantizeLinear_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, QuantizeLinear_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/quantizelinear.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* QuantizeLinear::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<QuantizeLinear, Layer>(m, "QuantizeLinear")
            .def("forward", &QuantizeLinear::forward);    
    }*/
}

#endif
