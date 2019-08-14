#ifndef SUM_H
#define SUM_H //Sum
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   sum_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Sum_parameter_descriptor{    
        
    };   

    struct Sum_input_desriptor{
        
        
    };

    struct Sum_output_descriptor{
        Tensor* sum_output;
        
    };

    struct Sum_binding_descriptor{
        
		
        
        
        Shape_t sum_output;
        
    };
}


namespace backend {

    class Sum : public Layer {
        Sum_parameter_descriptor parameters;
        Sum_input_desriptor      input;
        Sum_output_descriptor    output;
        Sum_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Sum_binding_descriptor>* program;
        
    public:
        Sum(std::string, Sum_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Sum() {}

    };
}

//cpp stuff
namespace backend {    
   
    Sum::Sum(std::string n, Sum_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Sum_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/sum.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Sum::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Sum, Layer>(m, "Sum")
            .def("forward", &Sum::forward);    
    }*/
}

#endif
