#ifndef ISNAN_H
#define ISNAN_H //IsNaN
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

    struct IsNaN_parameter_descriptor{    
        
    };   

    struct IsNaN_input_desriptor{
        Tensor* X_input;
        
    };

    struct IsNaN_output_descriptor{
        Tensor* Y_output;
        
    };

    struct IsNaN_binding_descriptor{
        
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class IsNaN : public Layer {
        IsNaN_parameter_descriptor parameters;
        IsNaN_input_desriptor      input;
        IsNaN_output_descriptor    output;
        IsNaN_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, IsNaN_binding_descriptor>* program;
        
    public:
        IsNaN(std::string, IsNaN_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~IsNaN() {}

    };
}

//cpp stuff
namespace backend {    
   
    IsNaN::IsNaN(std::string n, IsNaN_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, IsNaN_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/isnan.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* IsNaN::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<IsNaN, Layer>(m, "IsNaN")
            .def("forward", &IsNaN::forward);    
    }*/
}

#endif
