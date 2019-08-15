#ifndef MIN_H
#define MIN_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for min.
output: Output tensor.
//*/
//Min
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   min_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Min : public Layer {
        typedef struct {
            
			
            
            
            Shape_t min_output;
            
        } binding_descriptor;

        
        
        
        std::string min_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Min(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string min_output); 

        ~Min() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Min::Min(std::string n) : Layer(n) { }
       
    vuh::Device* Min::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Min::init() {      
    

		binding.min_output = tensor_dict[min_output]->shape();
 

    }
    
    void Min::call(std::string min_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/min.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[min_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Min, Layer>(m, "Min")
            .def(py::init<std::string> ())
            .def("forward", &Min::forward)
            .def("init", &Min::init)
            .def("call", (void (Min::*) (std::string)) &Min::call);
    }
}

#endif

/* PYTHON STUFF

*/

