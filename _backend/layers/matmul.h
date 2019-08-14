#ifndef MATMUL_H
#define MATMUL_H //MatMul
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct MatMul_parameter_descriptor{    
        
    };   

    struct MatMul_input_desriptor{
        Tensor* A_input; Tensor* B_input;
        
    };

    struct MatMul_output_descriptor{
        Tensor* Y_output;
        
    };

    struct MatMul_binding_descriptor{
        
		
        Shape_t A_input; Shape_t B_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class MatMul : public Layer {
        MatMul_parameter_descriptor parameters;
        MatMul_input_desriptor      input;
        MatMul_output_descriptor    output;
        MatMul_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, MatMul_binding_descriptor>* program;
        
    public:
        MatMul(std::string, MatMul_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~MatMul() {}

    };
}

//cpp stuff
namespace backend {    
   
    MatMul::MatMul(std::string n, MatMul_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, MatMul_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmul.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* MatMul::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<MatMul, Layer>(m, "MatMul")
            .def("forward", &MatMul::forward);    
    }*/
}

#endif
