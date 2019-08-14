#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H //LinearRegressor
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, intercepts, post_transform, targets
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct LinearRegressor_parameter_descriptor{    
        Tensor* coefficients; Tensor* intercepts; int post_transform; int targets;
    };   

    struct LinearRegressor_input_desriptor{
        Tensor* X_input;
        
    };

    struct LinearRegressor_output_descriptor{
        Tensor* Y_output;
        
    };

    struct LinearRegressor_binding_descriptor{
        int post_transform; int targets;
		Shape_t coefficients; Shape_t intercepts;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class LinearRegressor : public Layer {
        LinearRegressor_parameter_descriptor parameters;
        LinearRegressor_input_desriptor      input;
        LinearRegressor_output_descriptor    output;
        LinearRegressor_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, LinearRegressor_binding_descriptor>* program;
        
    public:
        LinearRegressor(std::string, LinearRegressor_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~LinearRegressor() {}

    };
}

//cpp stuff
namespace backend {    
   
    LinearRegressor::LinearRegressor(std::string n, LinearRegressor_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, LinearRegressor_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* LinearRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<LinearRegressor, Layer>(m, "LinearRegressor")
            .def("forward", &LinearRegressor::forward);    
    }*/
}

#endif
