#ifndef GRU_H
#define GRU_H //GRU
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, W_input, R_input
//OPTIONAL_INPUTS:          B_input_opt, sequence_lens_input_opt, initial_h_input_opt
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y_output_opt, Y_h_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size, linear_before_reset
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, Tensor*, float, int, int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct GRU_parameter_descriptor{    
        Tensor* activation_alpha; Tensor* activation_beta; Tensor* activations; float clip; int direction; int hidden_size; int linear_before_reset;
    };   

    struct GRU_input_desriptor{
        Tensor* X_input; Tensor* W_input; Tensor* R_input;
        Tensor* B_input_opt; Tensor* sequence_lens_input_opt; Tensor* initial_h_input_opt;
    };

    struct GRU_output_descriptor{
        
        Tensor* Y_output_opt; Tensor* Y_h_output_opt;
    };

    struct GRU_binding_descriptor{
        float clip; int direction; int hidden_size; int linear_before_reset;
		Shape_t activation_alpha; Shape_t activation_beta; Shape_t activations;
        Shape_t X_input; Shape_t W_input; Shape_t R_input;
        Shape_t B_input_opt; Shape_t sequence_lens_input_opt; Shape_t initial_h_input_opt;
        
        Shape_t Y_output_opt; Shape_t Y_h_output_opt;
    };
}


namespace backend {

    class GRU : public Layer {
        GRU_parameter_descriptor parameters;
        GRU_input_desriptor      input;
        GRU_output_descriptor    output;
        GRU_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, GRU_binding_descriptor>* program;
        
    public:
        GRU(std::string, GRU_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~GRU() {}

    };
}

//cpp stuff
namespace backend {    
   
    GRU::GRU(std::string n, GRU_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, GRU_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gru.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* GRU::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<GRU, Layer>(m, "GRU")
            .def("forward", &GRU::forward);    
    }*/
}

#endif
