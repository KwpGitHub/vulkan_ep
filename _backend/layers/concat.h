#ifndef CONCAT_H
#define CONCAT_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
Concatenate a list of tensors into a single tensor
input: List of tensors for concatenation
output: Concatenated tensor
//*/
//Concat
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   concat_result_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axis
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Concat : public Layer {
        typedef struct {
            int axis;
			
            
            
            Shape_t concat_result_output;
            
        } binding_descriptor;

        int axis;
        
        
        std::string concat_result_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Concat(std::string n, int axis);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string concat_result_output); 

        ~Concat() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Concat::Concat(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* Concat::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Concat::init() {      
    

		binding.concat_result_output = tensor_dict[concat_result_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void Concat::call(std::string concat_result_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/concat.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[concat_result_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Concat, Layer>(m, "Concat")
            .def(py::init<std::string, int> ())
            .def("forward", &Concat::forward)
            .def("init", &Concat::init)
            .def("call", (void (Concat::*) (std::string)) &Concat::call);
    }
}

#endif

/* PYTHON STUFF

*/

