#ifndef ATANH_H
#define ATANH_H //Atanh
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

    struct Atanh_parameter_descriptor{    
        
    };   

    struct Atanh_input_desriptor{
        Tensor* input_input;
        
    };

    struct Atanh_output_descriptor{
        Tensor* output_output;
        
    };

    struct Atanh_binding_descriptor{
        
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Atanh : public Layer {
        Atanh_parameter_descriptor parameters;
        Atanh_input_desriptor      input;
        Atanh_output_descriptor    output;
        Atanh_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Atanh_binding_descriptor>* program;
        
    public:
        Atanh(std::string, Atanh_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Atanh() {}

    };
}

//cpp stuff
namespace backend {    
   
    Atanh::Atanh(std::string n, Atanh_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Atanh_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/atanh.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Atanh::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Atanh, Layer>(m, "Atanh")
            .def("forward", &Atanh::forward);    
    }*/
}

#endif
