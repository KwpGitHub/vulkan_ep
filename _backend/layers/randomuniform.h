#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H //RandomUniform
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct RandomUniform_parameter_descriptor{    
        Shape_t shape; int dtype; float high; float low; float seed;
    };   

    struct RandomUniform_input_desriptor{
        
        
    };

    struct RandomUniform_output_descriptor{
        Tensor* output_output;
        
    };

    struct RandomUniform_binding_descriptor{
        Shape_t shape; int dtype; float high; float low; float seed;
		
        
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class RandomUniform : public Layer {
        RandomUniform_parameter_descriptor parameters;
        RandomUniform_input_desriptor      input;
        RandomUniform_output_descriptor    output;
        RandomUniform_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, RandomUniform_binding_descriptor>* program;
        
    public:
        RandomUniform(std::string, RandomUniform_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~RandomUniform() {}

    };
}

//cpp stuff
namespace backend {    
   
    RandomUniform::RandomUniform(std::string n, RandomUniform_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, RandomUniform_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniform.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* RandomUniform::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<RandomUniform, Layer>(m, "RandomUniform")
            .def("forward", &RandomUniform::forward);    
    }*/
}

#endif
