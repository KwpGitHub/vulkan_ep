#ifndef SIZE_H
#define SIZE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.

input: An input tensor.
output: Total number of elements of the input tensor
//*/
//Size
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   size_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Size : public Layer {
        typedef struct {
            
			
            Shape_t data_input;
            
            Shape_t size_output;
            
        } binding_descriptor;

        
        std::string data_input;
        
        std::string size_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Size(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string size_output); 

        ~Size() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Size::Size(std::string n) : Layer(n) { }
       
    vuh::Device* Size::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Size::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.size_output = tensor_dict[size_output]->shape();
 

    }
    
    void Size::call(std::string data_input, std::string size_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/size.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[size_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Size, Layer>(m, "Size")
            .def(py::init<std::string> ())
            .def("forward", &Size::forward)
            .def("init", &Size::init)
            .def("call", (void (Size::*) (std::string, std::string)) &Size::call);
    }
}

#endif

/* PYTHON STUFF

*/

