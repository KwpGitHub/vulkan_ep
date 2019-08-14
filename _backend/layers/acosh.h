#ifndef ACOSH_H
#define ACOSH_H //Acosh
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Acosh_parameter_descriptor{    
        
    };   

    struct Acosh_input_desriptor{
        Tensor* input_input;
        
    };

    struct Acosh_output_descriptor{
        Tensor* output_output;
        
    };

    struct Acosh_binding_descriptor{
        
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Acosh : public Layer {
        Acosh_parameter_descriptor parameters;
        Acosh_input_desriptor      input;
        Acosh_output_descriptor    output;
        Acosh_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Acosh_binding_descriptor>* program;
        
    public:
        Acosh(std::string, Acosh_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Acosh() {}

    };
}

//cpp stuff
namespace backend {    
   
    Acosh::Acosh(std::string n, Acosh_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Acosh_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/acosh.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Acosh::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Acosh, Layer>(m, "Acosh")
            .def("forward", &Acosh::forward);    
    }*/
}

#endif
