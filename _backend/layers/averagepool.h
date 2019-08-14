#ifndef AVERAGEPOOL_H
#define AVERAGEPOOL_H //AveragePool
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, ceil_mode, count_include_pad, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, int, Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct AveragePool_parameter_descriptor{    
        Shape_t kernel_shape; int auto_pad; int ceil_mode; int count_include_pad; Shape_t pads; Shape_t strides;
    };   

    struct AveragePool_input_desriptor{
        Tensor* X_input;
        
    };

    struct AveragePool_output_descriptor{
        Tensor* Y_output;
        
    };

    struct AveragePool_binding_descriptor{
        Shape_t kernel_shape; int auto_pad; int ceil_mode; int count_include_pad; Shape_t pads; Shape_t strides;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class AveragePool : public Layer {
        AveragePool_parameter_descriptor parameters;
        AveragePool_input_desriptor      input;
        AveragePool_output_descriptor    output;
        AveragePool_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, AveragePool_binding_descriptor>* program;
        
    public:
        AveragePool(std::string, AveragePool_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~AveragePool() {}

    };
}

//cpp stuff
namespace backend {    
   
    AveragePool::AveragePool(std::string n, AveragePool_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, AveragePool_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/averagepool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* AveragePool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<AveragePool, Layer>(m, "AveragePool")
            .def("forward", &AveragePool::forward);    
    }*/
}

#endif
