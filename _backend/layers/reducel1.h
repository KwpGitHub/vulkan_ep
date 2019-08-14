#ifndef REDUCEL1_H
#define REDUCEL1_H //ReduceL1
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

    struct ReduceL1_parameter_descriptor{    
        Shape_t axes; int keepdims;
    };   

    struct ReduceL1_input_desriptor{
        Tensor* data_input;
        
    };

    struct ReduceL1_output_descriptor{
        Tensor* reduced_output;
        
    };

    struct ReduceL1_binding_descriptor{
        Shape_t axes; int keepdims;
		
        Shape_t data_input;
        
        Shape_t reduced_output;
        
    };
}


namespace backend {

    class ReduceL1 : public Layer {
        ReduceL1_parameter_descriptor parameters;
        ReduceL1_input_desriptor      input;
        ReduceL1_output_descriptor    output;
        ReduceL1_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ReduceL1_binding_descriptor>* program;
        
    public:
        ReduceL1(std::string, ReduceL1_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ReduceL1() {}

    };
}

//cpp stuff
namespace backend {    
   
    ReduceL1::ReduceL1(std::string n, ReduceL1_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ReduceL1_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducel1.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ReduceL1::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceL1, Layer>(m, "ReduceL1")
            .def("forward", &ReduceL1::forward);    
    }*/
}

#endif
