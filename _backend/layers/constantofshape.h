#ifndef CONSTANTOFSHAPE_H
#define CONSTANTOFSHAPE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Generate a tensor with given value and shape.

input: 1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar.
output: Output tensor of shape specified by 'input'.If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype defaults to float32.
//*/
//ConstantOfShape
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      value
//OPTIONAL_PARAMETERS_TYPE: Tensor*

namespace py = pybind11;

//class stuff
namespace backend {   

    class ConstantOfShape : public Layer {
        typedef struct {
            
			Shape_t value;
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        std::string value;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ConstantOfShape(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string value, std::string input_input, std::string output_output); 

        ~ConstantOfShape() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ConstantOfShape::ConstantOfShape(std::string n) : Layer(n) { }
       
    vuh::Device* ConstantOfShape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ConstantOfShape::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.value = tensor_dict[value]->shape();
 
    }
    
    void ConstantOfShape::call(std::string value, std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constantofshape.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[value]->data(), *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ConstantOfShape, Layer>(m, "ConstantOfShape")
            .def(py::init<std::string> ())
            .def("forward", &ConstantOfShape::forward)
            .def("init", &ConstantOfShape::init)
            .def("call", (void (ConstantOfShape::*) (std::string, std::string, std::string)) &ConstantOfShape::call);
    }
}

#endif

/* PYTHON STUFF

*/

