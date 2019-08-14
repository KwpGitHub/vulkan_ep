#ifndef SELU_H
#define SELU_H //Selu
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, gamma
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Selu_parameter_descriptor{    
        float alpha; float gamma;
    };   

    struct Selu_input_desriptor{
        Tensor* X_input;
        
    };

    struct Selu_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Selu_binding_descriptor{
        float alpha; float gamma;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Selu : public Layer {
        Selu_parameter_descriptor parameters;
        Selu_input_desriptor      input;
        Selu_output_descriptor    output;
        Selu_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Selu_binding_descriptor>* program;
        
    public:
        Selu(std::string, Selu_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Selu() {}

    };
}

//cpp stuff
namespace backend {    
   
    Selu::Selu(std::string n, Selu_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Selu_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/selu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Selu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Selu, Layer>(m, "Selu")
            .def("forward", &Selu::forward);    
    }*/
}

#endif
