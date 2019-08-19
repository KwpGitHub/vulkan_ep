#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.

input: Input tensor with shape [batch_size, class_size], where class_size is the number of all possible outcomes. Each value along the axis zero represents the unnormalized log-probability of each corresponding outcome in a batch.
output: Output tensor with shape [batch_size, sample_size], where sample_size is the number of times to sample. Each value along the axis zero represents the outcome of the corresponding sample in a batch.
*/

//Multinomial
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, sample_size, seed
//OPTIONAL_PARAMETERS_TYPE: int, int, float

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
        Multinomial();
    
        void forward() { program->run(); }
        
        void init( int _dtype,  int _sample_size,  float _seed); 
        void bind(std::string _input_input, std::string _output_output); 

        ~Multinomial() {}
    };

    
    void init_layer_Multinomial(py::module& m) {
        // py::class_(m, "Multinomial");
    }
    

}


#endif

