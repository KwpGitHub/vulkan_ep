#include <vector>
namespace backend {
    class LinearRegressor {
    public:
        LinearRegressor ();
        ~LinearRegressor();
    private:

		std::vector<float> coefficients;
		std::vector<float> intercepts;
		std::string post_transform;
		int targets;
    };
}
