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
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_coefficients; std::vector<float> m_kernel_params; std::string m_kernel_type; int m_n_supports; int m_one_class; std::string m_post_transform; std::vector<float> m_rho; std::vector<float> m_support_vectors;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        SVMRegressor(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  int _n_supports,  int _one_class,  std::string _post_transform,  std::vector<float> _rho,  std::vector<float> _support_vectors); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~SVMRegressor() {}
    };
   
}
#endif

