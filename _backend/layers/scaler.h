#ifndef SCALER_H
#define SCALER_H //Scaler
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      offset, scale
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Scaler_parameter_descriptor{    
        Tensor* offset; Tensor* scale;
    };   

    struct Scaler_input_desriptor{
        Tensor* X_input;
        
    };

    struct Scaler_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Scaler_binding_descriptor{
        
		Shape_t offset; Shape_t scale;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Scaler : public Layer {
        Scaler_parameter_descriptor parameters;
        Scaler_input_desriptor      input;
        Scaler_output_descriptor    output;
        Scaler_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Scaler_binding_descriptor>* program;
        
    public:
        Scaler(std::string, Scaler_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Scaler() {}

    };
}

//cpp stuff
namespace backend {    
   
    Scaler::Scaler(std::string n, Scaler_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Scaler_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scaler.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Scaler::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Scaler, Layer>(m, "Scaler")
            .def("forward", &Scaler::forward);    
    }*/
}

#endif
