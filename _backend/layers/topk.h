#ifndef TOPK_H
#define TOPK_H //TopK
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, K_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Values_output, Indices_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct TopK_parameter_descriptor{    
        int axis;
    };   

    struct TopK_input_desriptor{
        Tensor* X_input; Tensor* K_input;
        
    };

    struct TopK_output_descriptor{
        Tensor* Values_output; Tensor* Indices_output;
        
    };

    struct TopK_binding_descriptor{
        int axis;
		
        Shape_t X_input; Shape_t K_input;
        
        Shape_t Values_output; Shape_t Indices_output;
        
    };
}


namespace backend {

    class TopK : public Layer {
        TopK_parameter_descriptor parameters;
        TopK_input_desriptor      input;
        TopK_output_descriptor    output;
        TopK_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, TopK_binding_descriptor>* program;
        
    public:
        TopK(std::string, TopK_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~TopK() {}

    };
}

//cpp stuff
namespace backend {    
   
    TopK::TopK(std::string n, TopK_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, TopK_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/topk.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* TopK::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<TopK, Layer>(m, "TopK")
            .def("forward", &TopK::forward);    
    }*/
}

#endif
