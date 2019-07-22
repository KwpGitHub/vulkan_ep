#include <vector>
namespace backend {
    class If {
    public:
        If ();
        ~If();
    private:

		graph else_branch;
		graph then_branch;
    };
}
