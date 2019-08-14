#ifndef MOD_H
#define MOD_H //Mod
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      fmod
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Mod_parameter_descriptor{    
        int fmod;
    };   

    struct Mod_input_desriptor{
        Tensor* A_input; Tensor* B_input;
        
    };

    struct Mod_output_descriptor{
        Tensor* C_output;
        
    };

    struct Mod_binding_descriptor{
        int fmod;
		
        Shape_t A_input; Shape_t B_input;
        
        Shape_t C_output;
        
    };
}


namespace backend {

    class Mod : public Layer {
        Mod_parameter_descriptor parameters;
        Mod_input_desriptor      input;
        Mod_output_descriptor    output;
        Mod_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Mod_binding_descriptor>* program;
        
    public:
        Mod(std::string, Mod_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Mod() {}

    };
}

//cpp stuff
namespace backend {    
   
    Mod::Mod(std::string n, Mod_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Mod_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mod.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Mod::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Mod, Layer>(m, "Mod")
            .def("forward", &Mod::forward);    
    }*/
}

#endif
