#ifndef RANDOMNORMAL_H
#define RANDOMNORMAL_H //RandomNormal
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct RandomNormal_parameter_descriptor{    
        Shape_t shape; int dtype; float mean; float scale; float seed;
    };   

    struct RandomNormal_input_desriptor{
        
        
    };

    struct RandomNormal_output_descriptor{
        Tensor* output_output;
        
    };

    struct RandomNormal_binding_descriptor{
        Shape_t shape; int dtype; float mean; float scale; float seed;
		
        
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class RandomNormal : public Layer {
        RandomNormal_parameter_descriptor parameters;
        RandomNormal_input_desriptor      input;
        RandomNormal_output_descriptor    output;
        RandomNormal_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, RandomNormal_binding_descriptor>* program;
        
    public:
        RandomNormal(std::string, RandomNormal_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~RandomNormal() {}

    };
}

//cpp stuff
namespace backend {    
   
    RandomNormal::RandomNormal(std::string n, RandomNormal_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, RandomNormal_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomnormal.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* RandomNormal::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<RandomNormal, Layer>(m, "RandomNormal")
            .def("forward", &RandomNormal::forward);    
    }*/
}

#endif
