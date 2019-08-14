#ifndef SLICE_H
#define SLICE_H //Slice
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input, starts_input, ends_input
//OPTIONAL_INPUTS:          axes_input_opt, steps_input_opt
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Slice_parameter_descriptor{    
        
    };   

    struct Slice_input_desriptor{
        Tensor* data_input; Tensor* starts_input; Tensor* ends_input;
        Tensor* axes_input_opt; Tensor* steps_input_opt;
    };

    struct Slice_output_descriptor{
        Tensor* output_output;
        
    };

    struct Slice_binding_descriptor{
        
		
        Shape_t data_input; Shape_t starts_input; Shape_t ends_input;
        Shape_t axes_input_opt; Shape_t steps_input_opt;
        Shape_t output_output;
        
    };
}


namespace backend {

    class Slice : public Layer {
        Slice_parameter_descriptor parameters;
        Slice_input_desriptor      input;
        Slice_output_descriptor    output;
        Slice_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Slice_binding_descriptor>* program;
        
    public:
        Slice(std::string, Slice_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Slice() {}

    };
}

//cpp stuff
namespace backend {    
   
    Slice::Slice(std::string n, Slice_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Slice_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/slice.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Slice::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Slice, Layer>(m, "Slice")
            .def("forward", &Slice::forward);    
    }*/
}

#endif
