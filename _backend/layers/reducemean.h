#ifndef REDUCEMEAN_H
#define REDUCEMEAN_H //ReduceMean
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

    struct ReduceMean_parameter_descriptor{    
        Shape_t axes; int keepdims;
    };   

    struct ReduceMean_input_desriptor{
        Tensor* data_input;
        
    };

    struct ReduceMean_output_descriptor{
        Tensor* reduced_output;
        
    };

    struct ReduceMean_binding_descriptor{
        Shape_t axes; int keepdims;
		
        Shape_t data_input;
        
        Shape_t reduced_output;
        
    };
}


namespace backend {

    class ReduceMean : public Layer {
        ReduceMean_parameter_descriptor parameters;
        ReduceMean_input_desriptor      input;
        ReduceMean_output_descriptor    output;
        ReduceMean_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ReduceMean_binding_descriptor>* program;
        
    public:
        ReduceMean(std::string, ReduceMean_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ReduceMean() {}

    };
}

//cpp stuff
namespace backend {    
   
    ReduceMean::ReduceMean(std::string n, ReduceMean_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ReduceMean_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducemean.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ReduceMean::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceMean, Layer>(m, "ReduceMean")
            .def("forward", &ReduceMean::forward);    
    }*/
}

#endif
