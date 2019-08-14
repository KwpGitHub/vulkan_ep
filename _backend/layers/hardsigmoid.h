#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H //HardSigmoid
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct HardSigmoid_parameter_descriptor{    
        float alpha; float beta;
    };   

    struct HardSigmoid_input_desriptor{
        Tensor* X_input;
        
    };

    struct HardSigmoid_output_descriptor{
        Tensor* Y_output;
        
    };

    struct HardSigmoid_binding_descriptor{
        float alpha; float beta;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class HardSigmoid : public Layer {
        HardSigmoid_parameter_descriptor parameters;
        HardSigmoid_input_desriptor      input;
        HardSigmoid_output_descriptor    output;
        HardSigmoid_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, HardSigmoid_binding_descriptor>* program;
        
    public:
        HardSigmoid(std::string, HardSigmoid_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~HardSigmoid() {}

    };
}

//cpp stuff
namespace backend {    
   
    HardSigmoid::HardSigmoid(std::string n, HardSigmoid_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, HardSigmoid_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/hardsigmoid.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* HardSigmoid::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<HardSigmoid, Layer>(m, "HardSigmoid")
            .def("forward", &HardSigmoid::forward);    
    }*/
}

#endif
