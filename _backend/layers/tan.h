#ifndef TAN_H
#define TAN_H //Tan
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

    struct Tan_parameter_descriptor{    
        
    };   

    struct Tan_input_desriptor{
        Tensor* input_input;
        
    };

    struct Tan_output_descriptor{
        Tensor* output_output;
        
    };

    struct Tan_binding_descriptor{
        
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Tan : public Layer {
        Tan_parameter_descriptor parameters;
        Tan_input_desriptor      input;
        Tan_output_descriptor    output;
        Tan_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Tan_binding_descriptor>* program;
        
    public:
        Tan(std::string, Tan_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Tan() {}

    };
}

//cpp stuff
namespace backend {    
   
    Tan::Tan(std::string n, Tan_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Tan_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/tan.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Tan::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Tan, Layer>(m, "Tan")
            .def("forward", &Tan::forward);    
    }*/
}

#endif
