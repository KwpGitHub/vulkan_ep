#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H 

#include "../layer.h"

/*

    Support Vector Machine regression prediction and one-class SVM anomaly detection.

input: Data to be regressed.
output: Regression outputs (one score per target per example).
*/

//SVMRegressor
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, kernel_params, kernel_type, n_supports, one_class, post_transform, rho, support_vectors
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<float>, std::string, int, int, std::string, std::vector<float>, std::vector<float>


//class stuff
namespace layers {   

    class SVMRegressor : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<float> coefficients; std::vector<float> kernel_params; std::string kernel_type; int n_supports; int one_class; std::string post_transform; std::vector<float> rho; std::vector<float> support_vectors;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        SVMRegressor(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init( std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  int _n_supports,  int _one_class,  std::string _post_transform,  std::vector<float> _rho,  std::vector<float> _support_vectors); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~SVMRegressor() {}
    };
   
}
#endif

