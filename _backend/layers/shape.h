#ifndef SHAPE_H
#define SHAPE_H 

#include "../layer.h"

/*

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.

input: An input tensor.
output: Shape of the input tensor
*/

//Shape
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   shape_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Shape : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i;
            
            backend::Shape_t shape_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        
        std::string data_i;
        
        std::string shape_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Shape(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _data_i, std::string _shape_o); 
        virtual void build();

        ~Shape() {}
    };
   
}
#endif

