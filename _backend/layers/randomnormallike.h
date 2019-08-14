#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H //RandomNormalLike
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct RandomNormalLike_parameter_descriptor{    
        int dtype; float mean; float scale; float seed;
    };   

    struct RandomNormalLike_input_desriptor{
        Tensor* input_input;
        
    };

    struct RandomNormalLike_output_descriptor{
        Tensor* output_output;
        
    };

    struct RandomNormalLike_binding_descriptor{
        int dtype; float mean; float scale; float seed;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class RandomNormalLike : public Layer {
        RandomNormalLike_parameter_descriptor parameters;
        RandomNormalLike_input_desriptor      input;
        RandomNormalLike_output_descriptor    output;
        RandomNormalLike_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, RandomNormalLike_binding_descriptor>* program;
        
    public:
        RandomNormalLike(std::string, RandomNormalLike_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~RandomNormalLike() {}

    };
}

//cpp stuff
namespace backend {    
   
    RandomNormalLike::RandomNormalLike(std::string n, RandomNormalLike_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, RandomNormalLike_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomnormallike.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* RandomNormalLike::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<RandomNormalLike, Layer>(m, "RandomNormalLike")
            .def("forward", &RandomNormalLike::forward);    
    }*/
}

#endif
