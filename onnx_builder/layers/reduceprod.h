#include <vector>
namespace backend {
    class ReduceProd {
    public:
        ReduceProd ();
        ~ReduceProd();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
