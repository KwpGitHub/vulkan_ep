#include <vector>
namespace backend {
    class SVMRegressor {
    public:
        SVMRegressor ();
        ~SVMRegressor();
    private:

		std::vector<float> coefficients;
		std::vector<float> kernel_params;
		std::string kernel_type;
		int n_supports;
		int one_class;
		std::string post_transform;
		std::vector<float> rho;
		std::vector<float> support_vectors;
    };
}
