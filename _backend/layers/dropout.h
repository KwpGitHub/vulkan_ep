#ifndef DROPOUT_H
#define DROPOUT_H //Dropout
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         mask_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      ratio
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Dropout_parameter_descriptor{    
        float ratio;
    };   

    struct Dropout_input_desriptor{
        Tensor* data_input;
        
    };

    struct Dropout_output_descriptor{
        Tensor* output_output;
        Tensor* mask_output_opt;
    };

    struct Dropout_binding_descriptor{
        float ratio;
		
        Shape_t data_input;
        
        Shape_t output_output;
        Shape_t mask_output_opt;
    };
}


namespace backend {

    class Dropout : public Layer {
        Dropout_parameter_descriptor parameters;
        Dropout_input_desriptor      input;
        Dropout_output_descriptor    output;
        Dropout_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Dropout_binding_descriptor>* program;
        
    public:
        Dropout(std::string, Dropout_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Dropout() {}

    };
}

//cpp stuff
namespace backend {    
   
    Dropout::Dropout(std::string n, Dropout_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Dropout_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/dropout.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Dropout::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Dropout, Layer>(m, "Dropout")
            .def("forward", &Dropout::forward);    
    }*/
}

#endif
