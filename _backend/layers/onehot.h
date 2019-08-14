#ifndef ONEHOT_H
#define ONEHOT_H //OneHot
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   indices_input, depth_input, values_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct OneHot_parameter_descriptor{    
        int axis;
    };   

    struct OneHot_input_desriptor{
        Tensor* indices_input; Tensor* depth_input; Tensor* values_input;
        
    };

    struct OneHot_output_descriptor{
        Tensor* output_output;
        
    };

    struct OneHot_binding_descriptor{
        int axis;
		
        Shape_t indices_input; Shape_t depth_input; Shape_t values_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class OneHot : public Layer {
        OneHot_parameter_descriptor parameters;
        OneHot_input_desriptor      input;
        OneHot_output_descriptor    output;
        OneHot_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, OneHot_binding_descriptor>* program;
        
    public:
        OneHot(std::string, OneHot_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~OneHot() {}

    };
}

//cpp stuff
namespace backend {    
   
    OneHot::OneHot(std::string n, OneHot_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, OneHot_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehot.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* OneHot::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<OneHot, Layer>(m, "OneHot")
            .def("forward", &OneHot::forward);    
    }*/
}

#endif
