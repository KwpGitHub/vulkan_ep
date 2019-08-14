#ifndef MATMULINTEGER_H
#define MATMULINTEGER_H //MatMulInteger
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          a_zero_point_input_opt, b_zero_point_input_opt
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct MatMulInteger_parameter_descriptor{    
        
    };   

    struct MatMulInteger_input_desriptor{
        Tensor* A_input; Tensor* B_input;
        Tensor* a_zero_point_input_opt; Tensor* b_zero_point_input_opt;
    };

    struct MatMulInteger_output_descriptor{
        Tensor* Y_output;
        
    };

    struct MatMulInteger_binding_descriptor{
        
		
        Shape_t A_input; Shape_t B_input;
        Shape_t a_zero_point_input_opt; Shape_t b_zero_point_input_opt;
        Shape_t Y_output;
        
    };
}


namespace backend {

    class MatMulInteger : public Layer {
        MatMulInteger_parameter_descriptor parameters;
        MatMulInteger_input_desriptor      input;
        MatMulInteger_output_descriptor    output;
        MatMulInteger_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, MatMulInteger_binding_descriptor>* program;
        
    public:
        MatMulInteger(std::string, MatMulInteger_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~MatMulInteger() {}

    };
}

//cpp stuff
namespace backend {    
   
    MatMulInteger::MatMulInteger(std::string n, MatMulInteger_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, MatMulInteger_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmulinteger.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* MatMulInteger::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<MatMulInteger, Layer>(m, "MatMulInteger")
            .def("forward", &MatMulInteger::forward);    
    }*/
}

#endif
