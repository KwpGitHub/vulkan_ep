#ifndef MIN_H
#define MIN_H //Min
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   min_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Min_parameter_descriptor{    
        
    };   

    struct Min_input_desriptor{
        
        
    };

    struct Min_output_descriptor{
        Tensor* min_output;
        
    };

    struct Min_binding_descriptor{
        
		
        
        
        Shape_t min_output;
        
    };
}


namespace backend {

    class Min : public Layer {
        Min_parameter_descriptor parameters;
        Min_input_desriptor      input;
        Min_output_descriptor    output;
        Min_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Min_binding_descriptor>* program;
        
    public:
        Min(std::string, Min_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Min() {}

    };
}

//cpp stuff
namespace backend {    
   
    Min::Min(std::string n, Min_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Min_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/min.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Min::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Min, Layer>(m, "Min")
            .def("forward", &Min::forward);    
    }*/
}

#endif
