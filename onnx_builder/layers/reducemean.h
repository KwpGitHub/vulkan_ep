#include <vector>
namespace backend {
    class ReduceMean {
    public:
        ReduceMean ();
        ~ReduceMean();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
