#ifndef MAX_H
#define MAX_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for max.
output: Output tensor.
//*/
//Max
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   max_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Max : public Layer {
        typedef struct {
            
			
            
            
            Shape_t max_output;
            
        } binding_descriptor;

        
        
        
        std::string max_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Max(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string max_output); 

        ~Max() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Max::Max(std::string n) : Layer(n) { }
       
    vuh::Device* Max::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Max::init() {      
    

		binding.max_output = tensor_dict[max_output]->shape();
 

    }
    
    void Max::call(std::string max_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/max.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[max_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Max, Layer>(m, "Max")
            .def(py::init<std::string> ())
            .def("forward", &Max::forward)
            .def("init", &Max::init)
            .def("call", (void (Max::*) (std::string)) &Max::call);
    }
}

#endif

/* PYTHON STUFF

*/

