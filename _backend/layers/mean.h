#ifndef MEAN_H
#define MEAN_H //Mean
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   mean_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Mean_parameter_descriptor{    
        
    };   

    struct Mean_input_desriptor{
        
        
    };

    struct Mean_output_descriptor{
        Tensor* mean_output;
        
    };

    struct Mean_binding_descriptor{
        
		
        
        
        Shape_t mean_output;
        
    };
}


namespace backend {

    class Mean : public Layer {
        Mean_parameter_descriptor parameters;
        Mean_input_desriptor      input;
        Mean_output_descriptor    output;
        Mean_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Mean_binding_descriptor>* program;
        
    public:
        Mean(std::string, Mean_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Mean() {}

    };
}

//cpp stuff
namespace backend {    
   
    Mean::Mean(std::string n, Mean_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Mean_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mean.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Mean::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Mean, Layer>(m, "Mean")
            .def("forward", &Mean::forward);    
    }*/
}

#endif
