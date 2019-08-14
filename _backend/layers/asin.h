#ifndef ASIN_H
#define ASIN_H //Asin
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Asin_parameter_descriptor{    
        
    };   

    struct Asin_input_desriptor{
        Tensor* input_input;
        
    };

    struct Asin_output_descriptor{
        Tensor* output_output;
        
    };

    struct Asin_binding_descriptor{
        
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Asin : public Layer {
        Asin_parameter_descriptor parameters;
        Asin_input_desriptor      input;
        Asin_output_descriptor    output;
        Asin_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Asin_binding_descriptor>* program;
        
    public:
        Asin(std::string, Asin_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Asin() {}

    };
}

//cpp stuff
namespace backend {    
   
    Asin::Asin(std::string n, Asin_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Asin_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/asin.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Asin::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Asin, Layer>(m, "Asin")
            .def("forward", &Asin::forward);    
    }*/
}

#endif
