#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H 

#include "../layer.h"

/*

    Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.

input: Data to be regressed.
output: Regression outputs (one per target, per example).
*/

//LinearRegressor
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, intercepts, post_transform, targets
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<float>, std::string, int


//class stuff
namespace layers {   

    class LinearRegressor : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_coefficients; std::vector<float> m_intercepts; std::string m_post_transform; int m_targets;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        LinearRegressor(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _coefficients,  std::vector<float> _intercepts,  std::string _post_transform,  int _targets); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~LinearRegressor() {}
    };
   
}
#endif

