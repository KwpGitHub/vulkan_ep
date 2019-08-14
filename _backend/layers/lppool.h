#ifndef LPPOOL_H
#define LPPOOL_H //LpPool
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, p, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct LpPool_parameter_descriptor{    
        Shape_t kernel_shape; int auto_pad; int p; Shape_t pads; Shape_t strides;
    };   

    struct LpPool_input_desriptor{
        Tensor* X_input;
        
    };

    struct LpPool_output_descriptor{
        Tensor* Y_output;
        
    };

    struct LpPool_binding_descriptor{
        Shape_t kernel_shape; int auto_pad; int p; Shape_t pads; Shape_t strides;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class LpPool : public Layer {
        LpPool_parameter_descriptor parameters;
        LpPool_input_desriptor      input;
        LpPool_output_descriptor    output;
        LpPool_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LpPool_binding_descriptor>* program;
        
    public:
        LpPool(std::string, LpPool_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LpPool() {}

    };
}

//cpp stuff
namespace backend {    
   
    LpPool::LpPool(std::string n, LpPool_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LpPool_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lppool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LpPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LpPool, Layer>(m, "LpPool")
            .def("forward", &LpPool::forward);    
    }*/
}

#endif
