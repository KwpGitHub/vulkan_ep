#ifndef REVERSESEQUENCE_H
#define REVERSESEQUENCE_H //ReverseSequence
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input, sequence_lens_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      batch_axis, time_axis
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ReverseSequence_parameter_descriptor{    
        int batch_axis; int time_axis;
    };   

    struct ReverseSequence_input_desriptor{
        Tensor* input_input; Tensor* sequence_lens_input;
        
    };

    struct ReverseSequence_output_descriptor{
        Tensor* Y_output;
        
    };

    struct ReverseSequence_binding_descriptor{
        int batch_axis; int time_axis;
		
        Shape_t input_input; Shape_t sequence_lens_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class ReverseSequence : public Layer {
        ReverseSequence_parameter_descriptor parameters;
        ReverseSequence_input_desriptor      input;
        ReverseSequence_output_descriptor    output;
        ReverseSequence_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ReverseSequence_binding_descriptor>* program;
        
    public:
        ReverseSequence(std::string, ReverseSequence_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ReverseSequence() {}

    };
}

//cpp stuff
namespace backend {    
   
    ReverseSequence::ReverseSequence(std::string n, ReverseSequence_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ReverseSequence_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reversesequence.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ReverseSequence::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ReverseSequence, Layer>(m, "ReverseSequence")
            .def("forward", &ReverseSequence::forward);    
    }*/
}

#endif
