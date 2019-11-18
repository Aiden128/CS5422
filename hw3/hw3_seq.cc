#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <limits>
#include <cstdint>

using namespace std;

const int32_t MaxDistance = INT16_MAX;
const int32_t Out_max = 1073741823;

class APSP{
private:
    int32_t num_vertex;
    std::vector< std::vector<int32_t> > AdjMatrix, Distance, Predecessor;
public:
    APSP():num_vertex(0){};
    APSP(int32_t n);
    inline void AddEdge(int32_t from, int32_t to, int32_t weight);
    void PrintData(std::vector< std::vector<int32_t> > array);
    void Write_file(const char *filename);
    void InitializeData();
    void FloydWarshall();
};

APSP::APSP(int32_t n):num_vertex(n){
    // Constructor, initialize AdjMatrix with 0 or MaxDistance
    AdjMatrix.resize(num_vertex);
    for (int32_t i = 0; i < num_vertex; i++) {
        AdjMatrix[i].resize(num_vertex, MaxDistance);
        for (int32_t j = 0; j < num_vertex; j++) {
            if (i == j){
                AdjMatrix[i][j] = 0;
            }
        }
    }
}

void APSP::InitializeData(){

    Distance.resize(num_vertex);
    Predecessor.resize(num_vertex);

    for (int32_t i = 0; i < num_vertex; i++) {
        Distance[i].resize(num_vertex);
        Predecessor[i].resize(num_vertex, -1);
        for (int32_t j = 0; j < num_vertex; j++) {
            Distance[i][j] = AdjMatrix[i][j];
            if (Distance[i][j] != 0 && Distance[i][j] != MaxDistance) {
                Predecessor[i][j] = i;
            }
        }
    }
}
void APSP::FloydWarshall(){

    InitializeData();

    // std::cout << "initial Distance[]:\n";
    // PrintData(Distance);
    // std::cout << "\ninitial Predecessor[]:\n";
    // PrintData(Predecessor);

    for (int32_t k = 0; k < num_vertex; k++) {
        for (int32_t i = 0; i < num_vertex; i++) {
            for (int32_t j = 0; j < num_vertex; j++) {
                if ((Distance[i][j] > Distance[i][k]+Distance[k][j]) 
                     && (Distance[i][k] != MaxDistance)) {
                    Distance[i][j] = Distance[i][k]+Distance[k][j];
                    Predecessor[i][j] = Predecessor[k][j];
                }
            }
        }
    }
    // print data after including new vertex and updating the shortest paths
    // std::cout << "Distance[]:\n";
    // PrintData(Distance);
    // std::cout << "\nPredecessor[]:\n";
    // PrintData(Predecessor);
}
void APSP::PrintData(std::vector< std::vector<int32_t> > array){

    for (int32_t i = 0; i < num_vertex; i++){
        for (int32_t j = 0; j < num_vertex; j++) {
            std::cout << std::setw(5) << array[i][j];
        }
        std::cout << std::endl;
    }
}

inline void APSP::AddEdge(int32_t from, int32_t to, int32_t weight){
    AdjMatrix[from][to] = weight;
}


void APSP::Write_file(const char *filename) {
    ofstream out_file(filename);
    for (int32_t i = 0; i < num_vertex; i++){
        for (int32_t j = 0; j < num_vertex; j++) {
            if(Distance[i][j] == INT16_MAX) {
                // Write 2^30 -1
                out_file.write((char*) &Out_max, sizeof(Out_max));
            } else {
                // Write normal
                out_file.write((char*) &Distance[i][j], sizeof(int32_t));
            }
        }
    }
    out_file.close();
}

int main(int argc, char **argv){

    std::ifstream file(argv[1]);
    int32_t num_vertex(0), num_edge(0);
    int32_t src(0), dest(0), weight(0);

    file.seekg(0, std::ios_base::beg);
    file.read( (char*)&num_vertex, sizeof(num_vertex));
    APSP apsp(num_vertex);
    file.read( (char*)&num_edge, sizeof(num_edge));
    for (int32_t i = 0; i < num_edge; ++i) {
        file.read( (char*)&src, sizeof(src));
        file.read( (char*)&dest, sizeof(dest));
        file.read( (char*)&weight, sizeof(weight));
        apsp.AddEdge(src, dest, weight);
    }
    file.close();
    
    // Check output
    // std::ifstream result(argv[2]);
    // vector<int> ref;
    // result.seekg(0, std::ios_base::beg);
    // int temp(0);
    // cout << "Reference: " << endl;
    // for(int i =0; i < (num_vertex * num_vertex); ++i) {
    //     result.read((char*)&temp, sizeof(temp));
    //     cout << temp << " ";
    //     if((i % num_vertex) == (num_vertex - 1)) {
    //         cout << endl;
    //     }
    //     ref.push_back(temp);
    // }
    // result.close();
    apsp.FloydWarshall();
    apsp.Write_file(argv[2]);
    return 0;
}