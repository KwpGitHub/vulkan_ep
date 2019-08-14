#ifndef SHRINK_H
#define SHRINK_H //Shrink
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      bias, lambd
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Shrink_parameter_descriptor{    
        float bias; float lambd;
    };   

    struct Shrink_input_desriptor{
        Tensor* input_input;
        
    };

    struct Shrink_output_descriptor{
        Tensor* output_output;
        
    };

    struct Shrink_binding_descriptor{
        float bias; float lambd;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Shrink : public Layer {
        Shrink_parameter_descriptor parameters;
        Shrink_input_desriptor      input;
        Shrink_output_descriptor    output;
        Shrink_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Shrink_binding_descriptor>* program;
        
    public:
        Shrink(std::string, Shrink_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Shrink() {}

    };
}

//cpp stuff
namespace backend {    
   
    Shrink::Shrink(std::string n, Shrink_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Shrink_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/shrink.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Shrink::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Shrink, Layer>(m, "Shrink")
            .def("forward", &Shrink::forward);    
    }*/
}

#endif
