#ifndef ISINF_H
#define ISINF_H //IsInf
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      detect_negative, detect_positive
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct IsInf_parameter_descriptor{    
        int detect_negative; int detect_positive;
    };   

    struct IsInf_input_desriptor{
        Tensor* X_input;
        
    };

    struct IsInf_output_descriptor{
        Tensor* Y_output;
        
    };

    struct IsInf_binding_descriptor{
        int detect_negative; int detect_positive;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class IsInf : public Layer {
        IsInf_parameter_descriptor parameters;
        IsInf_input_desriptor      input;
        IsInf_output_descriptor    output;
        IsInf_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, IsInf_binding_descriptor>* program;
        
    public:
        IsInf(std::string, IsInf_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~IsInf() {}

    };
}

//cpp stuff
namespace backend {    
   
    IsInf::IsInf(std::string n, IsInf_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, IsInf_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/isinf.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* IsInf::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<IsInf, Layer>(m, "IsInf")
            .def("forward", &IsInf::forward);    
    }*/
}

#endif
