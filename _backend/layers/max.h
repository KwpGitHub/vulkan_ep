#ifndef MAX_H
#define MAX_H 

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
//OUTPUS:                   max_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Max : public backend::Layer {
        typedef struct {          
            
            
            backend::Shape_t max_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        
        
        std::string max_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Max(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _max_o); 
        virtual void build();

        ~Max() {}
    };
   
}
#endif

