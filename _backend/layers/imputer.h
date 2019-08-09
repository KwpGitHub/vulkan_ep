#ifndef IMPUTER_H
#define IMPUTER_H //Imputer

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      imputed_value_floats, imputed_value_int64s, replaced_value_float, replaced_value_int64
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Shape_t, float, int



namespace backend {
    class Imputer : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
			Shape_t imputed_value_floats;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Imputer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Tensor* imputed_value_floats; Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
		Shape_t imputed_value_floats_t;
        //input
        std::string X_input;
        
        //output
        std::string Y_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~Imputer(){}
    };
}


namespace backend {    
    Imputer::Imputer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/imputer.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({imputed_value_int64s, replaced_value_float, replaced_value_int64, imputed_value_floats_t}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_input_o] );
    }

    vuh::Device* Imputer::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
