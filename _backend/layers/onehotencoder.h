#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H 

#include "../layer.h"

/*

    Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count 
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8, 
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.

input: Data to be encoded.
output: Encoded output data, having one more dimension than X.
*/

//OneHotEncoder
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, zeros
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, std::vector<std::string>, int


//class stuff
namespace layers {   

    class OneHotEncoder : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<int> cats_int64s; std::vector<std::string> cats_strings; int zeros;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        OneHotEncoder(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _cats_int64s,  std::vector<std::string> _cats_strings,  int _zeros); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~OneHotEncoder() {}
    };
   
}
#endif

