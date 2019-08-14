#ifndef RESHAPE_H
#define RESHAPE_H //Reshape
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input, shape_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reshaped_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Reshape_parameter_descriptor{    
        
    };   

    struct Reshape_input_desriptor{
        Tensor* data_input; Tensor* shape_input;
        
    };

    struct Reshape_output_descriptor{
        Tensor* reshaped_output;
        
    };

    struct Reshape_binding_descriptor{
        
		
        Shape_t data_input; Shape_t shape_input;
        
        Shape_t reshaped_output;
        
    };
}


namespace backend {

    class Reshape : public Layer {
        Reshape_parameter_descriptor parameters;
        Reshape_input_desriptor      input;
        Reshape_output_descriptor    output;
        Reshape_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Reshape_binding_descriptor>* program;
        
    public:
        Reshape(std::string, Reshape_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Reshape() {}

    };
}

//cpp stuff
namespace backend {    
   
    Reshape::Reshape(std::string n, Reshape_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Reshape_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reshape.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Reshape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Reshape, Layer>(m, "Reshape")
            .def("forward", &Reshape::forward);    
    }*/
}

#endif
