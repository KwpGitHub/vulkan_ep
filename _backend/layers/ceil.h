#ifndef CEIL_H
#define CEIL_H //Ceil
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
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

    struct Ceil_parameter_descriptor{    
        
    };   

    struct Ceil_input_desriptor{
        Tensor* X_input;
        
    };

    struct Ceil_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Ceil_binding_descriptor{
        
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Ceil : public Layer {
        Ceil_parameter_descriptor parameters;
        Ceil_input_desriptor      input;
        Ceil_output_descriptor    output;
        Ceil_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Ceil_binding_descriptor>* program;
        
    public:
        Ceil(std::string, Ceil_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Ceil() {}

    };
}

//cpp stuff
namespace backend {    
   
    Ceil::Ceil(std::string n, Ceil_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Ceil_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/ceil.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Ceil::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Ceil, Layer>(m, "Ceil")
            .def("forward", &Ceil::forward);    
    }*/
}

#endif
