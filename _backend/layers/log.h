#ifndef LOG_H
#define LOG_H //Log
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Log_parameter_descriptor{    
        
    };   

    struct Log_input_desriptor{
        Tensor* input_input;
        
    };

    struct Log_output_descriptor{
        Tensor* output_output;
        
    };

    struct Log_binding_descriptor{
        
		
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Log : public Layer {
        Log_parameter_descriptor parameters;
        Log_input_desriptor      input;
        Log_output_descriptor    output;
        Log_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Log_binding_descriptor>* program;
        
    public:
        Log(std::string, Log_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Log() {}

    };
}

//cpp stuff
namespace backend {    
   
    Log::Log(std::string n, Log_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Log_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/log.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Log::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Log, Layer>(m, "Log")
            .def("forward", &Log::forward);    
    }*/
}

#endif
