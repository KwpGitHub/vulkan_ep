#include <vector>
namespace backend {
    class ZipMap {
    public:
        ZipMap ();
        ~ZipMap();
    private:

		std::vector<int> classlabels_int64s;
		std::vector<string> classlabels_strings;
    };
}
