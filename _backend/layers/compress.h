#ifndef COMPRESS_H
#define COMPRESS_H //Compress
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input, condition_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Compress_parameter_descriptor{    
        int axis;
    };   

    struct Compress_input_desriptor{
        Tensor* input_input; Tensor* condition_input;
        
    };

    struct Compress_output_descriptor{
        Tensor* output_output;
        
    };

    struct Compress_binding_descriptor{
        int axis;
		
        Shape_t input_input; Shape_t condition_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Compress : public Layer {
        Compress_parameter_descriptor parameters;
        Compress_input_desriptor      input;
        Compress_output_descriptor    output;
        Compress_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Compress_binding_descriptor>* program;
        
    public:
        Compress(std::string, Compress_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Compress() {}

    };
}

//cpp stuff
namespace backend {    
   
    Compress::Compress(std::string n, Compress_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Compress_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/compress.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Compress::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Compress, Layer>(m, "Compress")
            .def("forward", &Compress::forward);    
    }*/
}

#endif
