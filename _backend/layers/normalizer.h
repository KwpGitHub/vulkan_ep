#ifndef NORMALIZER_H
#define NORMALIZER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.

input: Data to be encoded, a tensor of shape [N,C] or [C]
output: Encoded output data
//*/
//Normalizer
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      norm
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class Normalizer : public Layer {
        typedef struct {
            int norm;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int norm;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Normalizer(std::string n, int norm);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~Normalizer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Normalizer::Normalizer(std::string n, int norm) : Layer(n) { }
       
    vuh::Device* Normalizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Normalizer::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.norm = norm;
 
    }
    
    void Normalizer::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/normalizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Normalizer, Layer>(m, "Normalizer")
            .def(py::init<std::string, int> ())
            .def("forward", &Normalizer::forward)
            .def("init", &Normalizer::init)
            .def("call", (void (Normalizer::*) (std::string, std::string)) &Normalizer::call);
    }
}

#endif

/* PYTHON STUFF

*/

