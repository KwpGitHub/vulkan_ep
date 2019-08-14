#ifndef CONCAT_H
#define CONCAT_H //Concat
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   concat_result_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axis
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Concat_parameter_descriptor{    
        int axis;
    };   

    struct Concat_input_desriptor{
        
        
    };

    struct Concat_output_descriptor{
        Tensor* concat_result_output;
        
    };

    struct Concat_binding_descriptor{
        int axis;
		
        
        
        Shape_t concat_result_output;
        
    };
}


namespace backend {

    class Concat : public Layer {
        Concat_parameter_descriptor parameters;
        Concat_input_desriptor      input;
        Concat_output_descriptor    output;
        Concat_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Concat_binding_descriptor>* program;
        
    public:
        Concat(std::string, Concat_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Concat() {}

    };
}

//cpp stuff
namespace backend {    
   
    Concat::Concat(std::string n, Concat_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Concat_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/concat.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Concat::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Concat, Layer>(m, "Concat")
            .def("forward", &Concat::forward);    
    }*/
}

#endif
