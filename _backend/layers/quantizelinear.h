#ifndef QUANTIZELINEAR_H
#define QUANTIZELINEAR_H //QuantizeLinear

//INPUTS:                   x_input, y_scale_input
//OPTIONAL_INPUTS:          y_zero_point_input_o
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class QuantizeLinear : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t x_input; Shape_t y_scale_input;
            Shape_t y_zero_point_input_o;
            //output
            Shape_t y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        QuantizeLinear(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        
		
        //input
        std::string x_input; std::string y_scale_input;
        std::string y_zero_point_input_o;
        //output
        std::string y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~QuantizeLinear(){}
    };
}


namespace backend {    
    QuantizeLinear::QuantizeLinear(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/quantizelinear.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({tensor_dict[x_input]->shape(), tensor_dict[y_scale_input]->shape(), tensor_dict[y_zero_point_input_o]->shape(), tensor_dict[y_output]->shape()}, 
                            tensor_dict[x_input], tensor_dict[y_scale_input], tensor_dict[y_zero_point_input_o],
                            tensor_dict[y_output] );
    }

    vuh::Device* QuantizeLinear::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
