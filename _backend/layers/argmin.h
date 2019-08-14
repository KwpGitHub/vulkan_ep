#ifndef ARGMIN_H
#define ARGMIN_H //ArgMin
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, keepdims
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ArgMin_parameter_descriptor{    
        int axis; int keepdims;
    };   

    struct ArgMin_input_desriptor{
        Tensor* data_input;
        
    };

    struct ArgMin_output_descriptor{
        Tensor* reduced_output;
        
    };

    struct ArgMin_binding_descriptor{
        int axis; int keepdims;
		
        Shape_t data_input;
        
        Shape_t reduced_output;
        
    };
}


namespace backend {

    class ArgMin : public Layer {
        ArgMin_parameter_descriptor parameters;
        ArgMin_input_desriptor      input;
        ArgMin_output_descriptor    output;
        ArgMin_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ArgMin_binding_descriptor>* program;
        
    public:
        ArgMin(std::string, ArgMin_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ArgMin() {}

    };
}

//cpp stuff
namespace backend {    
   
    ArgMin::ArgMin(std::string n, ArgMin_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ArgMin_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/argmin.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ArgMin::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ArgMin, Layer>(m, "ArgMin")
            .def("forward", &ArgMin::forward);    
    }*/
}

#endif
