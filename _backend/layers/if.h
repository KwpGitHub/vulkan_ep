#ifndef IF_H
#define IF_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
namespace backend {   

    class If : public Layer {
        typedef struct {
            int else_branch; int then_branch;
			
            Shape_t cond_i;
            
            
            
        } binding_descriptor;

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

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/if.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[cond_i]->data());
        }

        ~If() {}
    };
   
}
#endif

