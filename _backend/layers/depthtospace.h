#ifndef DEPTHTOSPACE_H
#define DEPTHTOSPACE_H //DepthToSpace
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               blocksize
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct DepthToSpace_parameter_descriptor{    
        int blocksize;
    };   

    struct DepthToSpace_input_desriptor{
        Tensor* input_input;
        
    };

    struct DepthToSpace_output_descriptor{
        Tensor* output_output;
        
    };

    struct DepthToSpace_binding_descriptor{
        int blocksize;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class DepthToSpace : public Layer {
        DepthToSpace_parameter_descriptor parameters;
        DepthToSpace_input_desriptor      input;
        DepthToSpace_output_descriptor    output;
        DepthToSpace_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, DepthToSpace_binding_descriptor>* program;
        
    public:
        DepthToSpace(std::string, DepthToSpace_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~DepthToSpace() {}

    };
}

//cpp stuff
namespace backend {    
   
    DepthToSpace::DepthToSpace(std::string n, DepthToSpace_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, DepthToSpace_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/depthtospace.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* DepthToSpace::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<DepthToSpace, Layer>(m, "DepthToSpace")
            .def("forward", &DepthToSpace::forward);    
    }*/
}

#endif
