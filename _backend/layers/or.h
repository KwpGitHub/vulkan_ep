#ifndef OR_H
#define OR_H //Or
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Or_parameter_descriptor{    
        
    };   

    struct Or_input_desriptor{
        Tensor* A_input; Tensor* B_input;
        
    };

    struct Or_output_descriptor{
        Tensor* C_output;
        
    };

    struct Or_binding_descriptor{
        
		
        Shape_t A_input; Shape_t B_input;
        
        Shape_t C_output;
        
    };
}


namespace backend {

    class Or : public Layer {
        Or_parameter_descriptor parameters;
        Or_input_desriptor      input;
        Or_output_descriptor    output;
        Or_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Or_binding_descriptor>* program;
        
    public:
        Or(std::string, Or_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Or() {}

    };
}

//cpp stuff
namespace backend {    
   
    Or::Or(std::string n, Or_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Or_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/or.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Or::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Or, Layer>(m, "Or")
            .def("forward", &Or::forward);    
    }*/
}

#endif
