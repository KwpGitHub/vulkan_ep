#ifndef CLIP_H
#define CLIP_H //Clip
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      max, min
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Clip_parameter_descriptor{    
        float max; float min;
    };   

    struct Clip_input_desriptor{
        Tensor* input_input;
        
    };

    struct Clip_output_descriptor{
        Tensor* output_output;
        
    };

    struct Clip_binding_descriptor{
        float max; float min;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Clip : public Layer {
        Clip_parameter_descriptor parameters;
        Clip_input_desriptor      input;
        Clip_output_descriptor    output;
        Clip_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Clip_binding_descriptor>* program;
        
    public:
        Clip(std::string, Clip_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Clip() {}

    };
}

//cpp stuff
namespace backend {    
   
    Clip::Clip(std::string n, Clip_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Clip_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/clip.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Clip::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Clip, Layer>(m, "Clip")
            .def("forward", &Clip::forward);    
    }*/
}

#endif
