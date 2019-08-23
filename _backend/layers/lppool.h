#ifndef LPPOOL_H
#define LPPOOL_H 

#include "../layer.h"

/*

 LpPool consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
output: Output data tensor from Lp pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
*/

//LpPool
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          std::vector<int>
//OPTIONAL_PARAMETERS:      auto_pad, p, pads, strides
//OPTIONAL_PARAMETERS_TYPE: std::string, int, std::vector<int>, std::vector<int>


//class stuff
namespace layers {   

    class LpPool : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<int> kernel_shape; std::string auto_pad; int p; std::vector<int> pads; std::vector<int> strides;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        LpPool(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _kernel_shape,  std::string _auto_pad,  int _p,  std::vector<int> _pads,  std::vector<int> _strides); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~LpPool() {}
    };
   
}
#endif

