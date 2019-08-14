#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H //MaxRoiPool
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, rois_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pooled_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      spatial_scale
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct MaxRoiPool_parameter_descriptor{    
        Shape_t pooled_shape; float spatial_scale;
    };   

    struct MaxRoiPool_input_desriptor{
        Tensor* X_input; Tensor* rois_input;
        
    };

    struct MaxRoiPool_output_descriptor{
        Tensor* Y_output;
        
    };

    struct MaxRoiPool_binding_descriptor{
        Shape_t pooled_shape; float spatial_scale;
		
        Shape_t X_input; Shape_t rois_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class MaxRoiPool : public Layer {
        MaxRoiPool_parameter_descriptor parameters;
        MaxRoiPool_input_desriptor      input;
        MaxRoiPool_output_descriptor    output;
        MaxRoiPool_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, MaxRoiPool_binding_descriptor>* program;
        
    public:
        MaxRoiPool(std::string, MaxRoiPool_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~MaxRoiPool() {}

    };
}

//cpp stuff
namespace backend {    
   
    MaxRoiPool::MaxRoiPool(std::string n, MaxRoiPool_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, MaxRoiPool_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxroipool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* MaxRoiPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<MaxRoiPool, Layer>(m, "MaxRoiPool")
            .def("forward", &MaxRoiPool::forward);    
    }*/
}

#endif
