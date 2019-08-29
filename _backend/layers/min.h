#ifndef MIN_H
#define MIN_H 

#include "../layer.h"

/*

Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for min.
output: Output tensor.
*/

//Min
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   min_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Min : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        
        
        std::string min_o;
        

        binding_descriptor   binding;
       

    public:
        Min(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _min_o); 
        virtual void build();

        ~Min() {}
    };
   
}
#endif

