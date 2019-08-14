#ifndef SQUEEZE_H
#define SQUEEZE_H //Squeeze
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   squeezed_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Squeeze_parameter_descriptor{    
        Shape_t axes;
    };   

    struct Squeeze_input_desriptor{
        Tensor* data_input;
        
    };

    struct Squeeze_output_descriptor{
        Tensor* squeezed_output;
        
    };

    struct Squeeze_binding_descriptor{
        Shape_t axes;
		
        Shape_t data_input;
        
        Shape_t squeezed_output;
        
    };
}


namespace backend {

    class Squeeze : public Layer {
        Squeeze_parameter_descriptor parameters;
        Squeeze_input_desriptor      input;
        Squeeze_output_descriptor    output;
        Squeeze_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Squeeze_binding_descriptor>* program;
        
    public:
        Squeeze(std::string, Squeeze_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Squeeze() {}

    };
}

//cpp stuff
namespace backend {    
   
    Squeeze::Squeeze(std::string n, Squeeze_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Squeeze_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/squeeze.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Squeeze::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Squeeze, Layer>(m, "Squeeze")
            .def("forward", &Squeeze::forward);    
    }*/
}

#endif
