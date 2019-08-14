#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H //BatchNormalization
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, scale_input, B_input, mean_input, var_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         mean_output_opt, var_output_opt, saved_mean_output_opt, saved_var_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon, momentum
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct BatchNormalization_parameter_descriptor{    
        float epsilon; float momentum;
    };   

    struct BatchNormalization_input_desriptor{
        Tensor* X_input; Tensor* scale_input; Tensor* B_input; Tensor* mean_input; Tensor* var_input;
        
    };

    struct BatchNormalization_output_descriptor{
        Tensor* Y_output;
        Tensor* mean_output_opt; Tensor* var_output_opt; Tensor* saved_mean_output_opt; Tensor* saved_var_output_opt;
    };

    struct BatchNormalization_binding_descriptor{
        float epsilon; float momentum;
		
        Shape_t X_input; Shape_t scale_input; Shape_t B_input; Shape_t mean_input; Shape_t var_input;
        
        Shape_t Y_output;
        Shape_t mean_output_opt; Shape_t var_output_opt; Shape_t saved_mean_output_opt; Shape_t saved_var_output_opt;
    };
}


namespace backend {

    class BatchNormalization : public Layer {
        BatchNormalization_parameter_descriptor parameters;
        BatchNormalization_input_desriptor      input;
        BatchNormalization_output_descriptor    output;
        BatchNormalization_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, BatchNormalization_binding_descriptor>* program;
        
    public:
        BatchNormalization(std::string, BatchNormalization_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~BatchNormalization() {}

    };
}

//cpp stuff
namespace backend {    
   
    BatchNormalization::BatchNormalization(std::string n, BatchNormalization_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, BatchNormalization_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/batchnormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* BatchNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<BatchNormalization, Layer>(m, "BatchNormalization")
            .def("forward", &BatchNormalization::forward);    
    }*/
}

#endif
