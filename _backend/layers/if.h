#include "../layer.h"
#ifndef IF_H
#define IF_H 
/*
If conditional
input: Condition for the if
output: Values that are live-out to the enclosing scope. The return values in the `then_branch` and `else_branch` must be of the same shape and same data type.
//*/
//If
//INPUTS:                   cond_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               else_branch, then_branch
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class If : public Layer {
        typedef struct {
            int else_branch; int then_branch;
			
            Shape_t cond_input;
            
            
            
        } binding_descriptor;

        int else_branch; int then_branch;
        std::string cond_input;
        
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        If(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _else_branch,  int _then_branch); 
        void bind(std::string _cond_input); 

        ~If() {}

    };
    
}

#endif

