#ifndef PRELU_H
#define PRELU_H //PRelu
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, slope_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct PRelu_parameter_descriptor{    
        
    };   

    struct PRelu_input_desriptor{
        Tensor* X_input; Tensor* slope_input;
        
    };

    struct PRelu_output_descriptor{
        Tensor* Y_output;
        
    };

    struct PRelu_binding_descriptor{
        
		
        Shape_t X_input; Shape_t slope_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class PRelu : public Layer {
        PRelu_parameter_descriptor parameters;
        PRelu_input_desriptor      input;
        PRelu_output_descriptor    output;
        PRelu_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, PRelu_binding_descriptor>* program;
        
    public:
        PRelu(std::string, PRelu_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~PRelu() {}

    };
}

//cpp stuff
namespace backend {    
   
    PRelu::PRelu(std::string n, PRelu_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, PRelu_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/prelu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* PRelu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<PRelu, Layer>(m, "PRelu")
            .def("forward", &PRelu::forward);    
    }*/
}

#endif
