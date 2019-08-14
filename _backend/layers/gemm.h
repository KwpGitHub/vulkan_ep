#ifndef GEMM_H
#define GEMM_H //Gemm
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   A_input, B_input, C_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta, transA, transB
//OPTIONAL_PARAMETERS_TYPE: float, float, int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Gemm_parameter_descriptor{    
        float alpha; float beta; int transA; int transB;
    };   

    struct Gemm_input_desriptor{
        Tensor* A_input; Tensor* B_input; Tensor* C_input;
        
    };

    struct Gemm_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Gemm_binding_descriptor{
        float alpha; float beta; int transA; int transB;
		
        Shape_t A_input; Shape_t B_input; Shape_t C_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Gemm : public Layer {
        Gemm_parameter_descriptor parameters;
        Gemm_input_desriptor      input;
        Gemm_output_descriptor    output;
        Gemm_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Gemm_binding_descriptor>* program;
        
    public:
        Gemm(std::string, Gemm_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Gemm() {}

    };
}

//cpp stuff
namespace backend {    
   
    Gemm::Gemm(std::string n, Gemm_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Gemm_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gemm.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Gemm::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Gemm, Layer>(m, "Gemm")
            .def("forward", &Gemm::forward);    
    }*/
}

#endif
