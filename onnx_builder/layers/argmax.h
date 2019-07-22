#include <vector>
namespace backend {
    class ArgMax {
    public:
        ArgMax ();
        ~ArgMax();
    private:

		int axis;
		int keepdims;
    };
}
