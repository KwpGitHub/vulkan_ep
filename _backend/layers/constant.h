#ifndef CONSTANT_H
#define CONSTANT_H //Constant
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               value
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Constant_parameter_descriptor{    
        Tensor* value;
    };   

    struct Constant_input_desriptor{
        
        
    };

    struct Constant_output_descriptor{
        Tensor* output_output;
        
    };

    struct Constant_binding_descriptor{
        
		Shape_t value;
        
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Constant : public Layer {
        Constant_parameter_descriptor parameters;
        Constant_input_desriptor      input;
        Constant_output_descriptor    output;
        Constant_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Constant_binding_descriptor>* program;
        
    public:
        Constant(std::string, Constant_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Constant() {}

    };
}

//cpp stuff
namespace backend {    
   
    Constant::Constant(std::string n, Constant_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Constant_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constant.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Constant::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Constant, Layer>(m, "Constant")
            .def("forward", &Constant::forward);    
    }*/
}

#endif
