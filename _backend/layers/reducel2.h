#ifndef REDUCEL2_H
#define REDUCEL2_H //ReduceL2
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ReduceL2_parameter_descriptor{    
        Shape_t axes; int keepdims;
    };   

    struct ReduceL2_input_desriptor{
        Tensor* data_input;
        
    };

    struct ReduceL2_output_descriptor{
        Tensor* reduced_output;
        
    };

    struct ReduceL2_binding_descriptor{
        Shape_t axes; int keepdims;
		
        Shape_t data_input;
        
        Shape_t reduced_output;
        
    };
}


namespace backend {

    class ReduceL2 : public Layer {
        ReduceL2_parameter_descriptor parameters;
        ReduceL2_input_desriptor      input;
        ReduceL2_output_descriptor    output;
        ReduceL2_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ReduceL2_binding_descriptor>* program;
        
    public:
        ReduceL2(std::string, ReduceL2_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ReduceL2() {}

    };
}

//cpp stuff
namespace backend {    
   
    ReduceL2::ReduceL2(std::string n, ReduceL2_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ReduceL2_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducel2.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ReduceL2::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceL2, Layer>(m, "ReduceL2")
            .def("forward", &ReduceL2::forward);    
    }*/
}

#endif
