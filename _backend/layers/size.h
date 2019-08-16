#include "../layer.h"
#ifndef SIZE_H
#define SIZE_H 
/*

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.

input: An input tensor.
output: Total number of elements of the input tensor
//*/
//Size
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   size_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Size : public Layer {
        typedef struct {
            
			
            Shape_t data_input;
            
            Shape_t size_output;
            
        } binding_descriptor;

        
        std::string data_input;
        
        std::string size_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Size(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string size_output); 

        ~Size() {}

    };
    
}

#endif

