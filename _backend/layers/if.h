#ifndef IF_H
#define IF_H //If
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   cond_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               else_branch, then_branch
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct If_parameter_descriptor{    
        int else_branch; int then_branch;
    };   

    struct If_input_desriptor{
        Tensor* cond_input;
        
    };

    struct If_output_descriptor{
        
        
    };

    struct If_binding_descriptor{
        int else_branch; int then_branch;
		
        Shape_t cond_input;
        
        
        
    };
}


namespace backend {

    class If : public Layer {
        If_parameter_descriptor parameters;
        If_input_desriptor      input;
        If_output_descriptor    output;
        If_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, If_binding_descriptor>* program;
        
    public:
        If(std::string, If_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~If() {}

    };
}

//cpp stuff
namespace backend {    
   
    If::If(std::string n, If_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, If_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/if.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* If::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<If, Layer>(m, "If")
            .def("forward", &If::forward);    
    }*/
}

#endif
