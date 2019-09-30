#include "random.hpp"
#include <iostream>

using namespace std;

template <typename T> 
void oe_sort(vector<T>& lhs, vector<T> &rhs);

int main(void) {
    vector<int> rand_vec(random_gen<int>(10));

    cout<< "Random vector: " << endl;
    for (auto i : rand_vec){
        cout << i << " ";
    }
    cout<<endl;

    return 0;
}