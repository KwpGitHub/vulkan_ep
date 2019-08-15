#ifndef SUM_H
#define SUM_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for sum.
output: Output tensor.
//*/
//Sum
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   sum_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Sum : public Layer {
        typedef struct {
            
			
            
            
            Shape_t sum_output;
            
        } binding_descriptor;

        
        
        
        std::string sum_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sum(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string sum_output); 

        ~Sum() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Sum::Sum(std::string n) : Layer(n) { }
       
    vuh::Device* Sum::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Sum::init() {      
    

		binding.sum_output = tensor_dict[sum_output]->shape();
 

    }
    
    void Sum::call(std::string sum_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/sum.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[sum_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Sum, Layer>(m, "Sum")
            .def(py::init<std::string> ())
            .def("forward", &Sum::forward)
            .def("init", &Sum::init)
            .def("call", (void (Sum::*) (std::string)) &Sum::call);
    }
}

#endif

/* PYTHON STUFF

*/

