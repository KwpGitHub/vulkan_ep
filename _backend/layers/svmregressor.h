#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H //SVMRegressor
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, kernel_params, kernel_type, n_supports, one_class, post_transform, rho, support_vectors
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int, int, int, Tensor*, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct SVMRegressor_parameter_descriptor{    
        Tensor* coefficients; Tensor* kernel_params; int kernel_type; int n_supports; int one_class; int post_transform; Tensor* rho; Tensor* support_vectors;
    };   

    struct SVMRegressor_input_desriptor{
        Tensor* X_input;
        
    };

    struct SVMRegressor_output_descriptor{
        Tensor* Y_output;
        
    };

    struct SVMRegressor_binding_descriptor{
        int kernel_type; int n_supports; int one_class; int post_transform;
		Shape_t coefficients; Shape_t kernel_params; Shape_t rho; Shape_t support_vectors;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class SVMRegressor : public Layer {
        SVMRegressor_parameter_descriptor parameters;
        SVMRegressor_input_desriptor      input;
        SVMRegressor_output_descriptor    output;
        SVMRegressor_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, SVMRegressor_binding_descriptor>* program;
        
    public:
        SVMRegressor(std::string, SVMRegressor_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~SVMRegressor() {}

    };
}

//cpp stuff
namespace backend {    
   
    SVMRegressor::SVMRegressor(std::string n, SVMRegressor_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, SVMRegressor_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* SVMRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<SVMRegressor, Layer>(m, "SVMRegressor")
            .def("forward", &SVMRegressor::forward);    
    }*/
}

#endif
