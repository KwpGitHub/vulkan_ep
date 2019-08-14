#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H //OneHotEncoder
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, zeros
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct OneHotEncoder_parameter_descriptor{    
        Shape_t cats_int64s; Tensor* cats_strings; int zeros;
    };   

    struct OneHotEncoder_input_desriptor{
        Tensor* X_input;
        
    };

    struct OneHotEncoder_output_descriptor{
        Tensor* Y_output;
        
    };

    struct OneHotEncoder_binding_descriptor{
        Shape_t cats_int64s; int zeros;
		Shape_t cats_strings;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class OneHotEncoder : public Layer {
        OneHotEncoder_parameter_descriptor parameters;
        OneHotEncoder_input_desriptor      input;
        OneHotEncoder_output_descriptor    output;
        OneHotEncoder_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, OneHotEncoder_binding_descriptor>* program;
        
    public:
        OneHotEncoder(std::string, OneHotEncoder_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~OneHotEncoder() {}

    };
}

//cpp stuff
namespace backend {    
   
    OneHotEncoder::OneHotEncoder(std::string n, OneHotEncoder_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, OneHotEncoder_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehotencoder.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* OneHotEncoder::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<OneHotEncoder, Layer>(m, "OneHotEncoder")
            .def("forward", &OneHotEncoder::forward);    
    }*/
}

#endif
