#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H //LpNormalization
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, p
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct LpNormalization_parameter_descriptor{    
        int axis; int p;
    };   

    struct LpNormalization_input_desriptor{
        Tensor* input_input;
        
    };

    struct LpNormalization_output_descriptor{
        Tensor* output_output;
        
    };

    struct LpNormalization_binding_descriptor{
        int axis; int p;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class LpNormalization : public Layer {
        LpNormalization_parameter_descriptor parameters;
        LpNormalization_input_desriptor      input;
        LpNormalization_output_descriptor    output;
        LpNormalization_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LpNormalization_binding_descriptor>* program;
        
    public:
        LpNormalization(std::string, LpNormalization_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LpNormalization() {}

    };
}

//cpp stuff
namespace backend {    
   
    LpNormalization::LpNormalization(std::string n, LpNormalization_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LpNormalization_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lpnormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LpNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LpNormalization, Layer>(m, "LpNormalization")
            .def("forward", &LpNormalization::forward);    
    }*/
}

#endif
