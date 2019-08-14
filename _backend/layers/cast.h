#ifndef CAST_H
#define CAST_H //Cast
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               to
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Cast_parameter_descriptor{    
        int to;
    };   

    struct Cast_input_desriptor{
        Tensor* input_input;
        
    };

    struct Cast_output_descriptor{
        Tensor* output_output;
        
    };

    struct Cast_binding_descriptor{
        int to;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Cast : public Layer {
        Cast_parameter_descriptor parameters;
        Cast_input_desriptor      input;
        Cast_output_descriptor    output;
        Cast_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Cast_binding_descriptor>* program;
        
    public:
        Cast(std::string, Cast_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Cast() {}

    };
}

//cpp stuff
namespace backend {    
   
    Cast::Cast(std::string n, Cast_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Cast_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/cast.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Cast::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Cast, Layer>(m, "Cast")
            .def("forward", &Cast::forward);    
    }*/
}

#endif
