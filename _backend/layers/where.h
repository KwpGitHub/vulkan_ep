#ifndef WHERE_H
#define WHERE_H //Where
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   condition_input, X_input, Y_input
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

    struct Where_parameter_descriptor{    
        
    };   

    struct Where_input_desriptor{
        Tensor* condition_input; Tensor* X_input; Tensor* Y_input;
        
    };

    struct Where_output_descriptor{
        Tensor* output_output;
        
    };

    struct Where_binding_descriptor{
        
		
        Shape_t condition_input; Shape_t X_input; Shape_t Y_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Where : public Layer {
        Where_parameter_descriptor parameters;
        Where_input_desriptor      input;
        Where_output_descriptor    output;
        Where_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Where_binding_descriptor>* program;
        
    public:
        Where(std::string, Where_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Where() {}

    };
}

//cpp stuff
namespace backend {    
   
    Where::Where(std::string n, Where_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Where_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/where.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Where::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Where, Layer>(m, "Where")
            .def("forward", &Where::forward);    
    }*/
}

#endif
