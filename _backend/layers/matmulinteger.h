#ifndef MATMULINTEGER_H
#define MATMULINTEGER_H //MatMulInteger

#include "../layer.h"

//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          a_zero_point_input_o, b_zero_point_input_o
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class MatMulInteger : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t A_input; Shape_t B_input;
            Shape_t a_zero_point_input_o; Shape_t b_zero_point_input_o;
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        MatMulInteger(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        
		
        //input
        std::string A_input; std::string B_input;
        std::string a_zero_point_input_o; std::string b_zero_point_input_o;
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~MatMulInteger() {}
    };
}


namespace backend {    
    MatMulInteger::MatMulInteger(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/matmulinteger.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({tensor_dict[A_input]->shape(), tensor_dict[B_input]->shape(), tensor_dict[a_zero_point_input_o]->shape(), tensor_dict[b_zero_point_input_o]->shape(), tensor_dict[Y_output]->shape()} 
                        
                        , tensor_dict[A_input], tensor_dict[B_input], tensor_dict[a_zero_point_input_o], tensor_dict[b_zero_point_input_o], tensor_dict[Y_output] );
    }

    vuh::Device* MatMulInteger::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
