#include <vector>
namespace backend {
    class LabelEncoder {
    public:
        LabelEncoder ();
        ~LabelEncoder();
    private:

		float default_float;
		int default_int64;
		std::string default_string;
		std::vector<float> keys_floats;
		std::vector<int> keys_int64s;
		std::vector<string> keys_strings;
		std::vector<float> values_floats;
		std::vector<int> values_int64s;
		std::vector<string> values_strings;
    };
}
