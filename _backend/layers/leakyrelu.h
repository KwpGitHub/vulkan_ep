#ifndef LEAKYRELU_H
#define LEAKYRELU_H //LeakyRelu
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct LeakyRelu_parameter_descriptor{    
        float alpha;
    };   

    struct LeakyRelu_input_desriptor{
        Tensor* X_input;
        
    };

    struct LeakyRelu_output_descriptor{
        Tensor* Y_output;
        
    };

    struct LeakyRelu_binding_descriptor{
        float alpha;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class LeakyRelu : public Layer {
        LeakyRelu_parameter_descriptor parameters;
        LeakyRelu_input_desriptor      input;
        LeakyRelu_output_descriptor    output;
        LeakyRelu_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LeakyRelu_binding_descriptor>* program;
        
    public:
        LeakyRelu(std::string, LeakyRelu_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LeakyRelu() {}

    };
}

//cpp stuff
namespace backend {    
   
    LeakyRelu::LeakyRelu(std::string n, LeakyRelu_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LeakyRelu_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/leakyrelu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LeakyRelu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LeakyRelu, Layer>(m, "LeakyRelu")
            .def("forward", &LeakyRelu::forward);    
    }*/
}

#endif
