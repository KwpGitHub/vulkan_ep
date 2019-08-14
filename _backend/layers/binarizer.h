#ifndef BINARIZER_H
#define BINARIZER_H //Binarizer
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      threshold
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Binarizer_parameter_descriptor{    
        float threshold;
    };   

    struct Binarizer_input_desriptor{
        Tensor* X_input;
        
    };

    struct Binarizer_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Binarizer_binding_descriptor{
        float threshold;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Binarizer : public Layer {
        Binarizer_parameter_descriptor parameters;
        Binarizer_input_desriptor      input;
        Binarizer_output_descriptor    output;
        Binarizer_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Binarizer_binding_descriptor>* program;
        
    public:
        Binarizer(std::string, Binarizer_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Binarizer() {}

    };
}

//cpp stuff
namespace backend {    
   
    Binarizer::Binarizer(std::string n, Binarizer_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Binarizer_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/binarizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Binarizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Binarizer, Layer>(m, "Binarizer")
            .def("forward", &Binarizer::forward);    
    }*/
}

#endif
