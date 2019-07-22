#include <vector>
namespace backend {
    class CategoryMapper {
    public:
        CategoryMapper ();
        ~CategoryMapper();
    private:

		std::vector<int> cats_int64s;
		std::vector<string> cats_strings;
		int default_int64;
		std::string default_string;
    };
}
