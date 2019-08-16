#include "../layer.h"
#ifndef COMPRESS_H
#define COMPRESS_H 
/*

    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    
input: Tensor of rank r >= 1.
input: Rank 1 tensor of booleans to indicate which slices or data elements to be selected. Its length can be less than the input length alone the axis or the flattened input size if axis is not specified. In such cases data slices or elements exceeding the condition length are discarded.
output: Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.
//*/
//Compress
//INPUTS:                   input_input, condition_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

//class stuff
namespace backend {   

    class Compress : public Layer {
        typedef struct {
            int axis;
			
            Shape_t input_input; Shape_t condition_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis;
        std::string input_input; std::string condition_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Compress(std::string n, int axis);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string condition_input, std::string output_output); 

        ~Compress() {}

    };
    
}

#endif

