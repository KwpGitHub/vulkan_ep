#ifndef RECIPROCAL_H
#define RECIPROCAL_H //Reciprocal
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

    struct Reciprocal_parameter_descriptor{    
        
    };   

    struct Reciprocal_input_desriptor{
        Tensor* X_input;
        
    };

    struct Reciprocal_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Reciprocal_binding_descriptor{
        
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Reciprocal : public Layer {
        Reciprocal_parameter_descriptor parameters;
        Reciprocal_input_desriptor      input;
        Reciprocal_output_descriptor    output;
        Reciprocal_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Reciprocal_binding_descriptor>* program;
        
    public:
        Reciprocal(std::string, Reciprocal_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Reciprocal() {}

    };
}

//cpp stuff
namespace backend {    
   
    Reciprocal::Reciprocal(std::string n, Reciprocal_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Reciprocal_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reciprocal.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Reciprocal::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Reciprocal, Layer>(m, "Reciprocal")
            .def("forward", &Reciprocal::forward);    
    }*/
}

#endif
