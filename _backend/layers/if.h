#ifndef IF_H
#define IF_H 

#include "../layer.h"

/*
If conditional
input: Condition for the if
output: Values that are live-out to the enclosing scope. The return values in the `then_branch` and `else_branch` must be of the same shape and same data type.
*/

//If
//INPUTS:                   cond_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               else_branch, then_branch
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class If : public backend::Layer {
        typedef struct {          
            backend::Shape_t cond_i;
            
            
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        int else_branch; int then_branch;
        std::string cond_i;
        
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        If(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _else_branch,  int _then_branch); 
        virtual void bind(std::string _cond_i); 
        virtual void build();

        ~If() {}
    };
   
}
#endif

