#include "random.hpp"
#include <iostream>

using namespace std;

template <typename T>
inline void my_swap(T &a, T &b) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

template <typename T> 
const vector<T> oe_sort(const vector<T>& lhs){
    bool isSorted(false);
    vector<T> vec(lhs);

    while(!isSorted){
        isSorted = true;
        // Odd stage
        for(int i = 1; i < vec.size() - 1; i +=2){
            if(vec[i] > vec[i+1]){
                my_swap(vec[i], vec[i+1]);
                isSorted = false;
            }
        }
        // Even stage
        for(int i = 0; i < vec.size() - 1; i +=2){
            if(vec[i] > vec[i+1]){
                my_swap(vec[i], vec[i+1]);
                isSorted = false;
            }
        }
    }
    return vec;
}

int main(void) {
    vector<int> rand_vec(random_gen<int>(10));

    cout<< "Random vector: " << endl;
    for (auto i : rand_vec){
        cout << i << " ";
    }
    cout<<endl;
    cout << "Sorted vector:" << endl;
    vector<int> sorted_vec = oe_sort(rand_vec);
    for (auto i : sorted_vec){
        cout << i << " ";
    }
    cout << endl;

    return 0;
}