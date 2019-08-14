#ifndef PAD_H
#define PAD_H //Pad
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pads
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      mode, value
//OPTIONAL_PARAMETERS_TYPE: int, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Pad_parameter_descriptor{    
        Shape_t pads; int mode; float value;
    };   

    struct Pad_input_desriptor{
        Tensor* data_input;
        
    };

    struct Pad_output_descriptor{
        Tensor* output_output;
        
    };

    struct Pad_binding_descriptor{
        Shape_t pads; int mode; float value;
		
        Shape_t data_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Pad : public Layer {
        Pad_parameter_descriptor parameters;
        Pad_input_desriptor      input;
        Pad_output_descriptor    output;
        Pad_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Pad_binding_descriptor>* program;
        
    public:
        Pad(std::string, Pad_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Pad() {}

    };
}

//cpp stuff
namespace backend {    
   
    Pad::Pad(std::string n, Pad_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Pad_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pad.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Pad::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Pad, Layer>(m, "Pad")
            .def("forward", &Pad::forward);    
    }*/
}

#endif
