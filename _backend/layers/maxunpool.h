#ifndef MAXUNPOOL_H
#define MAXUNPOOL_H //MaxUnpool
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, I_input
//OPTIONAL_INPUTS:          output_shape_input_opt
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      pads, strides
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct MaxUnpool_parameter_descriptor{    
        Shape_t kernel_shape; Shape_t pads; Shape_t strides;
    };   

    struct MaxUnpool_input_desriptor{
        Tensor* X_input; Tensor* I_input;
        Tensor* output_shape_input_opt;
    };

    struct MaxUnpool_output_descriptor{
        Tensor* output_output;
        
    };

    struct MaxUnpool_binding_descriptor{
        Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        Shape_t X_input; Shape_t I_input;
        Shape_t output_shape_input_opt;
        Shape_t output_output;
        
    };
}


namespace backend {

    class MaxUnpool : public Layer {
        MaxUnpool_parameter_descriptor parameters;
        MaxUnpool_input_desriptor      input;
        MaxUnpool_output_descriptor    output;
        MaxUnpool_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, MaxUnpool_binding_descriptor>* program;
        
    public:
        MaxUnpool(std::string, MaxUnpool_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~MaxUnpool() {}

    };
}

//cpp stuff
namespace backend {    
   
    MaxUnpool::MaxUnpool(std::string n, MaxUnpool_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, MaxUnpool_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxunpool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* MaxUnpool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<MaxUnpool, Layer>(m, "MaxUnpool")
            .def("forward", &MaxUnpool::forward);    
    }*/
}

#endif
