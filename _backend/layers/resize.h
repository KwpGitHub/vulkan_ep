#ifndef RESIZE_H
#define RESIZE_H //Resize
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, scales_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      mode
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Resize_parameter_descriptor{    
        int mode;
    };   

    struct Resize_input_desriptor{
        Tensor* X_input; Tensor* scales_input;
        
    };

    struct Resize_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Resize_binding_descriptor{
        int mode;
		
        Shape_t X_input; Shape_t scales_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Resize : public Layer {
        Resize_parameter_descriptor parameters;
        Resize_input_desriptor      input;
        Resize_output_descriptor    output;
        Resize_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Resize_binding_descriptor>* program;
        
    public:
        Resize(std::string, Resize_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Resize() {}

    };
}

//cpp stuff
namespace backend {    
   
    Resize::Resize(std::string n, Resize_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Resize_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/resize.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Resize::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Resize, Layer>(m, "Resize")
            .def("forward", &Resize::forward);    
    }*/
}

#endif
