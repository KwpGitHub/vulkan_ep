#ifndef SPACETODEPTH_H
#define SPACETODEPTH_H //SpaceToDepth
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

    struct SpaceToDepth_parameter_descriptor{    
        int blocksize;
    };   

    struct SpaceToDepth_input_desriptor{
        Tensor* input_input;
        
    };

    struct SpaceToDepth_output_descriptor{
        Tensor* output_output;
        
    };

    struct SpaceToDepth_binding_descriptor{
        int blocksize;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class SpaceToDepth : public Layer {
        SpaceToDepth_parameter_descriptor parameters;
        SpaceToDepth_input_desriptor      input;
        SpaceToDepth_output_descriptor    output;
        SpaceToDepth_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, SpaceToDepth_binding_descriptor>* program;
        
    public:
        SpaceToDepth(std::string, SpaceToDepth_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~SpaceToDepth() {}

    };
}

//cpp stuff
namespace backend {    
   
    SpaceToDepth::SpaceToDepth(std::string n, SpaceToDepth_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, SpaceToDepth_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/spacetodepth.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* SpaceToDepth::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<SpaceToDepth, Layer>(m, "SpaceToDepth")
            .def("forward", &SpaceToDepth::forward);    
    }*/
}

#endif
