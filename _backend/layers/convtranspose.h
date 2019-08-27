#ifndef CONVTRANSPOSE_H
#define CONVTRANSPOSE_H 

#include "../layer.h"

/*

The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    
input: Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn)
input: The weight tensor that will be used in the convolutions; has size (C x M/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is the dimension of the kernel. The number of channels in the output should be equal to W.shape[1] * group (assuming zero based indices of the shape array)
input: Optional 1D bias to be added to the convolution, has size of M.
output: Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, pad lengths and group count. The number of channels in the output should be equal to W.shape[1] * group (assuming zero based indices of the shape array)
*/

//ConvTranspose
//INPUTS:                   X_i, W_i
//OPTIONAL_INPUTS:          B_i
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: std::string, std::vector<int>, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>


//class stuff
namespace layers {   

    class ConvTranspose : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i; backend::Shape_t W_i;
            backend::Shape_t B_i;
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string auto_pad; std::vector<int> dilations; int group; std::vector<int> kernel_shape; std::vector<int> output_padding; std::vector<int> output_shape; std::vector<int> pads; std::vector<int> strides;
        std::string X_i; std::string W_i;
        std::string B_i;
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        ConvTranspose(std::string name);
        
        virtual void forward();        
        virtual void init( std::string _auto_pad,  std::vector<int> _dilations,  int _group,  std::vector<int> _kernel_shape,  std::vector<int> _output_padding,  std::vector<int> _output_shape,  std::vector<int> _pads,  std::vector<int> _strides); 
        virtual void bind(std::string _X_i, std::string _W_i, std::string _B_i, std::string _Y_o); 
        virtual void build();

        ~ConvTranspose() {}
    };
   
}
#endif

