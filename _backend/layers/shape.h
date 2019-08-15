#ifndef SHAPE_H
#define SHAPE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.

input: An input tensor.
output: Shape of the input tensor
//*/
//Shape
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   shape_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Shape : public Layer {
        typedef struct {
            
			
            Shape_t data_input;
            
            Shape_t shape_output;
            
        } binding_descriptor;

        
        std::string data_input;
        
        std::string shape_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Shape(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string shape_output); 

        ~Shape() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Shape::Shape(std::string n) : Layer(n) { }
       
    vuh::Device* Shape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Shape::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.shape_output = tensor_dict[shape_output]->shape();
 

    }
    
    void Shape::call(std::string data_input, std::string shape_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/shape.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[shape_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Shape, Layer>(m, "Shape")
            .def(py::init<std::string> ())
            .def("forward", &Shape::forward)
            .def("init", &Shape::init)
            .def("call", (void (Shape::*) (std::string, std::string)) &Shape::call);
    }
}

#endif

/* PYTHON STUFF

*/

