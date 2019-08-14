#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H //InstanceNormalization
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input, scale_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct InstanceNormalization_parameter_descriptor{    
        float epsilon;
    };   

    struct InstanceNormalization_input_desriptor{
        Tensor* input_input; Tensor* scale_input; Tensor* B_input;
        
    };

    struct InstanceNormalization_output_descriptor{
        Tensor* output_output;
        
    };

    struct InstanceNormalization_binding_descriptor{
        float epsilon;
		
        Shape_t input_input; Shape_t scale_input; Shape_t B_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class InstanceNormalization : public Layer {
        InstanceNormalization_parameter_descriptor parameters;
        InstanceNormalization_input_desriptor      input;
        InstanceNormalization_output_descriptor    output;
        InstanceNormalization_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, InstanceNormalization_binding_descriptor>* program;
        
    public:
        InstanceNormalization(std::string, InstanceNormalization_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~InstanceNormalization() {}

    };
}

//cpp stuff
namespace backend {    
   
    InstanceNormalization::InstanceNormalization(std::string n, InstanceNormalization_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, InstanceNormalization_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/instancenormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* InstanceNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<InstanceNormalization, Layer>(m, "InstanceNormalization")
            .def("forward", &InstanceNormalization::forward);    
    }*/
}

#endif
