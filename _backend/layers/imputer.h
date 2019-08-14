#ifndef IMPUTER_H
#define IMPUTER_H //Imputer
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      imputed_value_floats, imputed_value_int64s, replaced_value_float, replaced_value_int64
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Shape_t, float, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Imputer_parameter_descriptor{    
        Tensor* imputed_value_floats; Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
    };   

    struct Imputer_input_desriptor{
        Tensor* X_input;
        
    };

    struct Imputer_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Imputer_binding_descriptor{
        Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
		Shape_t imputed_value_floats;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Imputer : public Layer {
        Imputer_parameter_descriptor parameters;
        Imputer_input_desriptor      input;
        Imputer_output_descriptor    output;
        Imputer_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Imputer_binding_descriptor>* program;
        
    public:
        Imputer(std::string, Imputer_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Imputer() {}

    };
}

//cpp stuff
namespace backend {    
   
    Imputer::Imputer(std::string n, Imputer_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Imputer_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/imputer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Imputer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Imputer, Layer>(m, "Imputer")
            .def("forward", &Imputer::forward);    
    }*/
}

#endif
