#include <vector>
namespace backend {
    class OneHotEncoder {
    public:
        OneHotEncoder ();
        ~OneHotEncoder();
    private:

		std::vector<int> cats_int64s;
		std::vector<string> cats_strings;
		int zeros;
    };
}
