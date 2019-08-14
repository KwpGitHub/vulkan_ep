#ifndef REDUCELOGSUMEXP_H
#define REDUCELOGSUMEXP_H //ReduceLogSumExp
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ReduceLogSumExp_parameter_descriptor{    
        Shape_t axes; int keepdims;
    };   

    struct ReduceLogSumExp_input_desriptor{
        Tensor* data_input;
        
    };

    struct ReduceLogSumExp_output_descriptor{
        Tensor* reduced_output;
        
    };

    struct ReduceLogSumExp_binding_descriptor{
        Shape_t axes; int keepdims;
		
        Shape_t data_input;
        
        Shape_t reduced_output;
        
    };
}


namespace backend {

    class ReduceLogSumExp : public Layer {
        ReduceLogSumExp_parameter_descriptor parameters;
        ReduceLogSumExp_input_desriptor      input;
        ReduceLogSumExp_output_descriptor    output;
        ReduceLogSumExp_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ReduceLogSumExp_binding_descriptor>* program;
        
    public:
        ReduceLogSumExp(std::string, ReduceLogSumExp_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ReduceLogSumExp() {}

    };
}

//cpp stuff
namespace backend {    
   
    ReduceLogSumExp::ReduceLogSumExp(std::string n, ReduceLogSumExp_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ReduceLogSumExp_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducelogsumexp.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ReduceLogSumExp::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ReduceLogSumExp, Layer>(m, "ReduceLogSumExp")
            .def("forward", &ReduceLogSumExp::forward);    
    }*/
}

#endif
