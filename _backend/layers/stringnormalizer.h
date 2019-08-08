#ifndef STRINGNORMALIZER_H
#define STRINGNORMALIZER_H //StringNormalizer

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      case_change_action, is_case_sensitive, locale, stopwords
//OPTIONAL_PARAMETERS_TYPE: int, int, int, Tensor*



namespace backend {
    class StringNormalizer : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int case_change_action; int is_case_sensitive; int locale;
			Shape_t stopwords;
            //input
            Shape_t X;
            
            //output
            Shape_t Y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        StringNormalizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int case_change_action; int is_case_sensitive; int locale; Tensor* stopwords;
		Shape_t stopwords;
        //input
        std::string X;
        
        //output
        std::string Y;
        
        //std::vector<uint32_t> output_shape();
   
        ~StringNormalizer(){}
    };
}


namespace backend {    
    StringNormalizer::StringNormalizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/stringnormalizer.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* StringNormalizer::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
