#include <vector>
namespace backend {
    class LSTM {
    public:
        LSTM ();
        ~LSTM();
    private:

		std::vector<float> activation_alpha;
		std::vector<float> activation_beta;
		std::vector<string> activations;
		float clip;
		std::string direction;
		int hidden_size;
		int input_forget;
    };
}
