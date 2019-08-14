#ifndef RELU_H
#define RELU_H //Relu
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

    struct Relu_parameter_descriptor{    
        
    };   

    struct Relu_input_desriptor{
        Tensor* X_input;
        
    };

    struct Relu_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Relu_binding_descriptor{
        
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Relu : public Layer {
        Relu_parameter_descriptor parameters;
        Relu_input_desriptor      input;
        Relu_output_descriptor    output;
        Relu_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Relu_binding_descriptor>* program;
        
    public:
        Relu(std::string, Relu_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Relu() {}

    };
}

//cpp stuff
namespace backend {    
   
    Relu::Relu(std::string n, Relu_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Relu_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/relu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Relu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Relu, Layer>(m, "Relu")
            .def("forward", &Relu::forward);    
    }*/
}

#endif
