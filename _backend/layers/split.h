#ifndef SPLIT_H
#define SPLIT_H //Split
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, split
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Split_parameter_descriptor{    
        int axis; Shape_t split;
    };   

    struct Split_input_desriptor{
        Tensor* input_input;
        
    };

    struct Split_output_descriptor{
        
        
    };

    struct Split_binding_descriptor{
        int axis; Shape_t split;
		
        Shape_t input_input;
        
        
        
    };
}


namespace backend {

    class Split : public Layer {
        Split_parameter_descriptor parameters;
        Split_input_desriptor      input;
        Split_output_descriptor    output;
        Split_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Split_binding_descriptor>* program;
        
    public:
        Split(std::string, Split_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Split() {}

    };
}

//cpp stuff
namespace backend {    
   
    Split::Split(std::string n, Split_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Split_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/split.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Split::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Split, Layer>(m, "Split")
            .def("forward", &Split::forward);    
    }*/
}

#endif
