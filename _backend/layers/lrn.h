#ifndef LRN_H
#define LRN_H //LRN
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               size
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      alpha, beta, bias
//OPTIONAL_PARAMETERS_TYPE: float, float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct LRN_parameter_descriptor{    
        int size; float alpha; float beta; float bias;
    };   

    struct LRN_input_desriptor{
        Tensor* X_input;
        
    };

    struct LRN_output_descriptor{
        Tensor* Y_output;
        
    };

    struct LRN_binding_descriptor{
        int size; float alpha; float beta; float bias;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class LRN : public Layer {
        LRN_parameter_descriptor parameters;
        LRN_input_desriptor      input;
        LRN_output_descriptor    output;
        LRN_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LRN_binding_descriptor>* program;
        
    public:
        LRN(std::string, LRN_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LRN() {}

    };
}

//cpp stuff
namespace backend {    
   
    LRN::LRN(std::string n, LRN_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LRN_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lrn.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LRN::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LRN, Layer>(m, "LRN")
            .def("forward", &LRN::forward);    
    }*/
}

#endif
