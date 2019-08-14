#ifndef SQRT_H
#define SQRT_H //Sqrt
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

    struct Sqrt_parameter_descriptor{    
        
    };   

    struct Sqrt_input_desriptor{
        Tensor* X_input;
        
    };

    struct Sqrt_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Sqrt_binding_descriptor{
        
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Sqrt : public Layer {
        Sqrt_parameter_descriptor parameters;
        Sqrt_input_desriptor      input;
        Sqrt_output_descriptor    output;
        Sqrt_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Sqrt_binding_descriptor>* program;
        
    public:
        Sqrt(std::string, Sqrt_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Sqrt() {}

    };
}

//cpp stuff
namespace backend {    
   
    Sqrt::Sqrt(std::string n, Sqrt_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Sqrt_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/sqrt.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Sqrt::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Sqrt, Layer>(m, "Sqrt")
            .def("forward", &Sqrt::forward);    
    }*/
}

#endif
