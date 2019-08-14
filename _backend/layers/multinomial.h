#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H //Multinomial
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, sample_size, seed
//OPTIONAL_PARAMETERS_TYPE: int, int, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Multinomial_parameter_descriptor{    
        int dtype; int sample_size; float seed;
    };   

    struct Multinomial_input_desriptor{
        Tensor* input_input;
        
    };

    struct Multinomial_output_descriptor{
        Tensor* output_output;
        
    };

    struct Multinomial_binding_descriptor{
        int dtype; int sample_size; float seed;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Multinomial : public Layer {
        Multinomial_parameter_descriptor parameters;
        Multinomial_input_desriptor      input;
        Multinomial_output_descriptor    output;
        Multinomial_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Multinomial_binding_descriptor>* program;
        
    public:
        Multinomial(std::string, Multinomial_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Multinomial() {}

    };
}

//cpp stuff
namespace backend {    
   
    Multinomial::Multinomial(std::string n, Multinomial_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Multinomial_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/multinomial.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Multinomial::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Multinomial, Layer>(m, "Multinomial")
            .def("forward", &Multinomial::forward);    
    }*/
}

#endif
