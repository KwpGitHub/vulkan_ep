#ifndef SHAPE_H
#define SHAPE_H //Shape
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   shape_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Shape_parameter_descriptor{    
        
    };   

    struct Shape_input_desriptor{
        Tensor* data_input;
        
    };

    struct Shape_output_descriptor{
        Tensor* shape_output;
        
    };

    struct Shape_binding_descriptor{
        
		
        Shape_t data_input;
        
        Shape_t shape_output;
        
    };
}


namespace backend {

    class Shape : public Layer {
        Shape_parameter_descriptor parameters;
        Shape_input_desriptor      input;
        Shape_output_descriptor    output;
        Shape_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Shape_binding_descriptor>* program;
        
    public:
        Shape(std::string, Shape_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Shape() {}

    };
}

//cpp stuff
namespace backend {    
   
    Shape::Shape(std::string n, Shape_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Shape_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/shape.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Shape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Shape, Layer>(m, "Shape")
            .def("forward", &Shape::forward);    
    }*/
}

#endif
