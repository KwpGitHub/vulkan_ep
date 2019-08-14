#ifndef TRANSPOSE_H
#define TRANSPOSE_H //Transpose
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   transposed_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      perm
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Transpose_parameter_descriptor{    
        Shape_t perm;
    };   

    struct Transpose_input_desriptor{
        Tensor* data_input;
        
    };

    struct Transpose_output_descriptor{
        Tensor* transposed_output;
        
    };

    struct Transpose_binding_descriptor{
        Shape_t perm;
		
        Shape_t data_input;
        
        Shape_t transposed_output;
        
    };
}


namespace backend {

    class Transpose : public Layer {
        Transpose_parameter_descriptor parameters;
        Transpose_input_desriptor      input;
        Transpose_output_descriptor    output;
        Transpose_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Transpose_binding_descriptor>* program;
        
    public:
        Transpose(std::string, Transpose_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Transpose() {}

    };
}

//cpp stuff
namespace backend {    
   
    Transpose::Transpose(std::string n, Transpose_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Transpose_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/transpose.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Transpose::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Transpose, Layer>(m, "Transpose")
            .def("forward", &Transpose::forward);    
    }*/
}

#endif
