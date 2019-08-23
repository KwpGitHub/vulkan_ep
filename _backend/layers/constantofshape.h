#ifndef CONSTANTOFSHAPE_H
#define CONSTANTOFSHAPE_H 

#include "../layer.h"

/*

Generate a tensor with given value and shape.

input: 1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar.
output: Output tensor of shape specified by 'input'.If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype defaults to float32.
*/

//ConstantOfShape
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      value
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>


//class stuff
namespace layers {   

    class ConstantOfShape : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<float> value;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        ConstantOfShape(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<float> _value); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~ConstantOfShape() {}
    };
   
}
#endif

