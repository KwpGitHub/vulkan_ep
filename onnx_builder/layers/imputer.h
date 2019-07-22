#include <vector>
namespace backend {
    class Imputer {
    public:
        Imputer ();
        ~Imputer();
    private:

		std::vector<float> imputed_value_floats;
		std::vector<int> imputed_value_int64s;
		float replaced_value_float;
		int replaced_value_int64;
    };
}
