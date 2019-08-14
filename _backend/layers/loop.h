#ifndef LOOP_H
#define LOOP_H //Loop
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          M_input_opt, cond_input_opt
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               body
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Loop_parameter_descriptor{    
        int body;
    };   

    struct Loop_input_desriptor{
        
        Tensor* M_input_opt; Tensor* cond_input_opt;
    };

    struct Loop_output_descriptor{
        
        
    };

    struct Loop_binding_descriptor{
        int body;
		
        
        Shape_t M_input_opt; Shape_t cond_input_opt;
        
        
    };
}


namespace backend {

    class Loop : public Layer {
        Loop_parameter_descriptor parameters;
        Loop_input_desriptor      input;
        Loop_output_descriptor    output;
        Loop_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Loop_binding_descriptor>* program;
        
    public:
        Loop(std::string, Loop_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Loop() {}

    };
}

//cpp stuff
namespace backend {    
   
    Loop::Loop(std::string n, Loop_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Loop_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/loop.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Loop::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Loop, Layer>(m, "Loop")
            .def("forward", &Loop::forward);    
    }*/
}

#endif
