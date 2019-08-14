#ifndef POW_H
#define POW_H //Pow
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Pow_parameter_descriptor{    
        
    };   

    struct Pow_input_desriptor{
        Tensor* X_input; Tensor* Y_input;
        
    };

    struct Pow_output_descriptor{
        Tensor* Z_output;
        
    };

    struct Pow_binding_descriptor{
        
		
        Shape_t X_input; Shape_t Y_input;
        
        Shape_t Z_output;
        
    };
}


namespace backend {

    class Pow : public Layer {
        Pow_parameter_descriptor parameters;
        Pow_input_desriptor      input;
        Pow_output_descriptor    output;
        Pow_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Pow_binding_descriptor>* program;
        
    public:
        Pow(std::string, Pow_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Pow() {}

    };
}

//cpp stuff
namespace backend {    
   
    Pow::Pow(std::string n, Pow_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Pow_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pow.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Pow::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Pow, Layer>(m, "Pow")
            .def("forward", &Pow::forward);    
    }*/
}

#endif
