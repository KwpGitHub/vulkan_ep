#include <vector>
namespace backend {
    class GRU {
    public:
        GRU ();
        ~GRU();
    private:

		std::vector<float> activation_alpha;
		std::vector<float> activation_beta;
		std::vector<string> activations;
		float clip;
		std::string direction;
		int hidden_size;
		int linear_before_reset;
    };
}
