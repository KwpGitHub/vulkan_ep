#ifndef MAX_H
#define MAX_H //Max
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   max_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Max_parameter_descriptor{    
        
    };   

    struct Max_input_desriptor{
        
        
    };

    struct Max_output_descriptor{
        Tensor* max_output;
        
    };

    struct Max_binding_descriptor{
        
		
        
        
        Shape_t max_output;
        
    };
}


namespace backend {

    class Max : public Layer {
        Max_parameter_descriptor parameters;
        Max_input_desriptor      input;
        Max_output_descriptor    output;
        Max_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Max_binding_descriptor>* program;
        
    public:
        Max(std::string, Max_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Max() {}

    };
}

//cpp stuff
namespace backend {    
   
    Max::Max(std::string n, Max_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Max_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/max.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Max::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Max, Layer>(m, "Max")
            .def("forward", &Max::forward);    
    }*/
}

#endif
