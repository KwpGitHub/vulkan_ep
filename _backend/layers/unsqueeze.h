#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H //Unsqueeze
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   expanded_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axes
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Unsqueeze_parameter_descriptor{    
        Shape_t axes;
    };   

    struct Unsqueeze_input_desriptor{
        Tensor* data_input;
        
    };

    struct Unsqueeze_output_descriptor{
        Tensor* expanded_output;
        
    };

    struct Unsqueeze_binding_descriptor{
        Shape_t axes;
		
        Shape_t data_input;
        
        Shape_t expanded_output;
        
    };
}


namespace backend {

    class Unsqueeze : public Layer {
        Unsqueeze_parameter_descriptor parameters;
        Unsqueeze_input_desriptor      input;
        Unsqueeze_output_descriptor    output;
        Unsqueeze_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Unsqueeze_binding_descriptor>* program;
        
    public:
        Unsqueeze(std::string, Unsqueeze_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Unsqueeze() {}

    };
}

//cpp stuff
namespace backend {    
   
    Unsqueeze::Unsqueeze(std::string n, Unsqueeze_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Unsqueeze_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/unsqueeze.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Unsqueeze::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Unsqueeze, Layer>(m, "Unsqueeze")
            .def("forward", &Unsqueeze::forward);    
    }*/
}

#endif
