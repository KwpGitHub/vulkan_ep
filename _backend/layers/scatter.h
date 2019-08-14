#ifndef SCATTER_H
#define SCATTER_H //Scatter
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input, indices_input, updates_input
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

    struct Scatter_parameter_descriptor{    
        int axis;
    };   

    struct Scatter_input_desriptor{
        Tensor* data_input; Tensor* indices_input; Tensor* updates_input;
        
    };

    struct Scatter_output_descriptor{
        Tensor* output_output;
        
    };

    struct Scatter_binding_descriptor{
        int axis;
		
        Shape_t data_input; Shape_t indices_input; Shape_t updates_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class Scatter : public Layer {
        Scatter_parameter_descriptor parameters;
        Scatter_input_desriptor      input;
        Scatter_output_descriptor    output;
        Scatter_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Scatter_binding_descriptor>* program;
        
    public:
        Scatter(std::string, Scatter_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Scatter() {}

    };
}

//cpp stuff
namespace backend {    
   
    Scatter::Scatter(std::string n, Scatter_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Scatter_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scatter.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Scatter::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Scatter, Layer>(m, "Scatter")
            .def("forward", &Scatter::forward);    
    }*/
}

#endif
