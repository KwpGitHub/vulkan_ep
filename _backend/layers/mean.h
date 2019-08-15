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

*/
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
            
        } parameter_descriptor;  

        typedef struct {
            
            
        } input_desriptor;

        typedef struct {
            Tensor* mean_output;
            
        } output_descriptor;

        typedef struct {
            
		
            
            
            Shape_t mean_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Mean(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Mean() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Mean::Mean(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mean.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Mean::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Mean::init() {

		binding.mean_output = output.mean_output->shape();
 

        program->bind(binding, *output.mean_output->data());
    }
    
    void Mean::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Mean, Layer>(m, "Mean")
            .def("forward", &Mean::forward);    
    }
}*/

#endif
