#ifndef GLOBALLPPOOL_H
#define GLOBALLPPOOL_H //GlobalLpPool
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      p
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct GlobalLpPool_parameter_descriptor{    
        int p;
    };   

    struct GlobalLpPool_input_desriptor{
        Tensor* X_input;
        
    };

    struct GlobalLpPool_output_descriptor{
        Tensor* Y_output;
        
    };

    struct GlobalLpPool_binding_descriptor{
        int p;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class GlobalLpPool : public Layer {
        GlobalLpPool_parameter_descriptor parameters;
        GlobalLpPool_input_desriptor      input;
        GlobalLpPool_output_descriptor    output;
        GlobalLpPool_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, GlobalLpPool_binding_descriptor>* program;
        
    public:
        GlobalLpPool(std::string, GlobalLpPool_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~GlobalLpPool() {}

    };
}

//cpp stuff
namespace backend {    
   
    GlobalLpPool::GlobalLpPool(std::string n, GlobalLpPool_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, GlobalLpPool_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/globallppool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* GlobalLpPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<GlobalLpPool, Layer>(m, "GlobalLpPool")
            .def("forward", &GlobalLpPool::forward);    
    }*/
}

#endif
