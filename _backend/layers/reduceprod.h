#ifndef REDUCEPROD_H
#define REDUCEPROD_H //ReduceProd
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

    struct ReduceProd_parameter_descriptor{    
        Shape_t axes; int keepdims;
    };   

    struct ReduceProd_input_desriptor{
        Tensor* data_input;
        
    };

    struct ReduceProd_output_descriptor{
        Tensor* reduced_output;
        
    };

    struct ReduceProd_binding_descriptor{
        Shape_t axes; int keepdims;
		
        Shape_t data_input;
        
        Shape_t reduced_output;
        
    };
}


namespace backend {

    class ReduceProd : public Layer {
        ReduceProd_parameter_descriptor parameters;
        ReduceProd_input_desriptor      input;
        ReduceProd_output_descriptor    output;
        ReduceProd_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ReduceProd_binding_descriptor>* program;
        
    public:
        ReduceProd(std::string, ReduceProd_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ReduceProd() {}

    };
}

//cpp stuff
namespace backend {    
   
    ReduceProd::ReduceProd(std::string n, ReduceProd_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ReduceProd_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reduceprod.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ReduceProd::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceProd, Layer>(m, "ReduceProd")
            .def("forward", &ReduceProd::forward);    
    }*/
}

#endif
