#ifndef LOGSOFTMAX_H
#define LOGSOFTMAX_H //LogSoftmax
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
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

    struct LogSoftmax_parameter_descriptor{    
        int axis;
    };   

    struct LogSoftmax_input_desriptor{
        Tensor* input_input;
        
    };

    struct LogSoftmax_output_descriptor{
        Tensor* output_output;
        
    };

    struct LogSoftmax_binding_descriptor{
        int axis;
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class LogSoftmax : public Layer {
        LogSoftmax_parameter_descriptor parameters;
        LogSoftmax_input_desriptor      input;
        LogSoftmax_output_descriptor    output;
        LogSoftmax_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LogSoftmax_binding_descriptor>* program;
        
    public:
        LogSoftmax(std::string, LogSoftmax_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LogSoftmax() {}

    };
}

//cpp stuff
namespace backend {    
   
    LogSoftmax::LogSoftmax(std::string n, LogSoftmax_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LogSoftmax_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/logsoftmax.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LogSoftmax::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LogSoftmax, Layer>(m, "LogSoftmax")
            .def("forward", &LogSoftmax::forward);    
    }*/
}

#endif
