#ifndef MAXPOOL_H
#define MAXPOOL_H //MaxPool
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         Indices_output_opt
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, ceil_mode, dilations, pads, storage_order, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t, int, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct MaxPool_parameter_descriptor{    
        Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
    };   

    struct MaxPool_input_desriptor{
        Tensor* X_input;
        
    };

    struct MaxPool_output_descriptor{
        Tensor* Y_output;
        Tensor* Indices_output_opt;
    };

    struct MaxPool_binding_descriptor{
        Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        Shape_t Indices_output_opt;
    };
}


namespace backend {

    class MaxPool : public Layer {
        MaxPool_parameter_descriptor parameters;
        MaxPool_input_desriptor      input;
        MaxPool_output_descriptor    output;
        MaxPool_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, MaxPool_binding_descriptor>* program;
        
    public:
        MaxPool(std::string, MaxPool_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~MaxPool() {}

    };
}

//cpp stuff
namespace backend {    
   
    MaxPool::MaxPool(std::string n, MaxPool_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, MaxPool_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxpool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* MaxPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<MaxPool, Layer>(m, "MaxPool")
            .def("forward", &MaxPool::forward);    
    }*/
}

#endif
