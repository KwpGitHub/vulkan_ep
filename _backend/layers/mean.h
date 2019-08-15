#ifndef MEAN_H
#define MEAN_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for mean.
output: Output tensor.
//*/
//Mean
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   mean_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Mean : public Layer {
        typedef struct {
            
			
            
            
            Shape_t mean_output;
            
        } binding_descriptor;

        
        
        
        std::string mean_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Mean(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string mean_output); 

        ~Mean() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Mean::Mean(std::string n) : Layer(n) { }
       
    vuh::Device* Mean::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Mean::init() {      
    

		binding.mean_output = tensor_dict[mean_output]->shape();
 

    }
    
    void Mean::call(std::string mean_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mean.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[mean_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Mean, Layer>(m, "Mean")
            .def(py::init<std::string> ())
            .def("forward", &Mean::forward)
            .def("init", &Mean::init)
            .def("call", (void (Mean::*) (std::string)) &Mean::call);
    }
}

#endif

/* PYTHON STUFF

*/

