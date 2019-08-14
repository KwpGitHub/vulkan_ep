#ifndef EYELIKE_H
#define EYELIKE_H //EyeLike
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, k
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct EyeLike_parameter_descriptor{    
        int dtype; int k;
    };   

    struct EyeLike_input_desriptor{
        Tensor* input_input;
        
    };

    struct EyeLike_output_descriptor{
        Tensor* output_output;
        
    };

    struct EyeLike_binding_descriptor{
        int dtype; int k;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class EyeLike : public Layer {
        EyeLike_parameter_descriptor parameters;
        EyeLike_input_desriptor      input;
        EyeLike_output_descriptor    output;
        EyeLike_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, EyeLike_binding_descriptor>* program;
        
    public:
        EyeLike(std::string, EyeLike_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~EyeLike() {}

    };
}

//cpp stuff
namespace backend {    
   
    EyeLike::EyeLike(std::string n, EyeLike_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, EyeLike_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/eyelike.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* EyeLike::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<EyeLike, Layer>(m, "EyeLike")
            .def("forward", &EyeLike::forward);    
    }*/
}

#endif
