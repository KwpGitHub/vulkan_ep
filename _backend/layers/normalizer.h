#ifndef NORMALIZER_H
#define NORMALIZER_H //Normalizer
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      norm
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Normalizer_parameter_descriptor{    
        int norm;
    };   

    struct Normalizer_input_desriptor{
        Tensor* X_input;
        
    };

    struct Normalizer_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Normalizer_binding_descriptor{
        int norm;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Normalizer : public Layer {
        Normalizer_parameter_descriptor parameters;
        Normalizer_input_desriptor      input;
        Normalizer_output_descriptor    output;
        Normalizer_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Normalizer_binding_descriptor>* program;
        
    public:
        Normalizer(std::string, Normalizer_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Normalizer() {}

    };
}

//cpp stuff
namespace backend {    
   
    Normalizer::Normalizer(std::string n, Normalizer_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Normalizer_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/normalizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Normalizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Normalizer, Layer>(m, "Normalizer")
            .def("forward", &Normalizer::forward);    
    }*/
}

#endif
