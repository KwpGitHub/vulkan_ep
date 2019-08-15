#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.

input: Input tensor with shape [batch_size, class_size], where class_size is the number of all possible outcomes. Each value along the axis zero represents the unnormalized log-probability of each corresponding outcome in a batch.
output: Output tensor with shape [batch_size, sample_size], where sample_size is the number of times to sample. Each value along the axis zero represents the outcome of the corresponding sample in a batch.
//*/
//Multinomial
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, sample_size, seed
//OPTIONAL_PARAMETERS_TYPE: int, int, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Multinomial : public Layer {
        typedef struct {
            int dtype; int sample_size; float seed;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int dtype; int sample_size; float seed;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Multinomial(std::string n, int dtype, int sample_size, float seed);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~Multinomial() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Multinomial::Multinomial(std::string n, int dtype, int sample_size, float seed) : Layer(n) { }
       
    vuh::Device* Multinomial::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Multinomial::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.dtype = dtype;
  		binding.sample_size = sample_size;
  		binding.seed = seed;
 
    }
    
    void Multinomial::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/multinomial.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Multinomial, Layer>(m, "Multinomial")
            .def(py::init<std::string, int, int, float> ())
            .def("forward", &Multinomial::forward)
            .def("init", &Multinomial::init)
            .def("call", (void (Multinomial::*) (std::string, std::string)) &Multinomial::call);
    }
}

#endif

/* PYTHON STUFF

*/

