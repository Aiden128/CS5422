#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

void print(vector<int> &data, const int &n);

int main(void) {
    int num_vertex(5);
    int num_edge(num_vertex * (num_vertex - 1));
    vector<int> temp;
    ofstream file("testcase.in");

    file.write((char *)&num_vertex, sizeof(int));
    file.write((char *)&num_edge, sizeof(int));

    for (int i = 0; i < num_vertex; ++i) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<int> distribution(0, 1000);
        for (int j = 0; j < num_vertex; ++j) {
            if (i == j) {
                continue;
            }
            int weight(distribution(generator));
            // temp.push_back(i);
            // temp.push_back(j);
            // temp.push_back(weight);
            file.write((char *)&i, sizeof(int));
            file.write((char *)&j, sizeof(int));
            file.write((char *)&weight, sizeof(int));
        }
    }
    // print(temp, num_vertex);
    file.close();
    return 0;
}

void print(vector<int> &data, const int &n) {
    for (int i = 0; i < n * (n - 1); ++i) {
        int src(data[i * 3]);
        int dest(data[i * 3 + 1]);
        int weight(data[i * 3 + 2]);
        cout << "Edge: " << src << ", " << dest << ", " << weight << endl;
    }
}
