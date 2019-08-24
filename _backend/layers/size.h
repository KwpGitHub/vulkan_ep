#ifndef SIZE_H
#define SIZE_H 

#include "../layer.h"

/*

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.

input: An input tensor.
output: Total number of elements of the input tensor
*/

//Size
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   size_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Size : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i;
            
            backend::Shape_t size_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        std::string data_i;
        
        std::string size_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Size(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _data_i, std::string _size_o); 
        virtual void build();

        ~Size() {}
    };
   
}
#endif

