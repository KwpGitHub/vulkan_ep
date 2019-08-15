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

*/
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
            
        } parameter_descriptor;  

        typedef struct {
            
            
        } input_desriptor;

        typedef struct {
            Tensor* max_output;
            
        } output_descriptor;

        typedef struct {
            
		
            
            
            Shape_t max_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Max(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Max() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Max::Max(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/max.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Max::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Max::init() {

		binding.max_output = output.max_output->shape();
 

        program->bind(binding, *output.max_output->data());
    }
    
    void Max::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Max, Layer>(m, "Max")
            .def("forward", &Max::forward);    
    }
}*/

#endif
