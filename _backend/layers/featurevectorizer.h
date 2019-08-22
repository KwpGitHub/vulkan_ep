#ifndef FEATUREVECTORIZER_H
#define FEATUREVECTORIZER_H 

#include "../layer.h"

/*

    Concatenates input tensors into one continuous output.<br>
    All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
    Inputs are copied to the output maintaining the order of the input arguments.<br>
    All inputs must be integers or floats, while the output will be all floating point values.

input: An ordered collection of tensors, all with the same element type.
output: The output array, elements ordered as the inputs.
*/

//FeatureVectorizer
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      inputdimensions
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>


//class stuff
namespace layers {   

    class FeatureVectorizer : public backend::Layer {
        typedef struct {          
            
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        std::vector<int> inputdimensions;
        
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        FeatureVectorizer(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _inputdimensions); 
        virtual void bind(std::string _Y_o); 
        virtual void build();

        ~FeatureVectorizer() {}
    };
   
}
#endif

