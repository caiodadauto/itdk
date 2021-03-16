#include <iostream>
#include <numeric>
#include <vector>
#include "_paths.h"


void dfs_to_target(int source, int target, const std::vector<std::vector<float>>& WU, std::vector<short>& visited, std::vector<std::vector<float>>& paths, std::vector<float>& path)
{
    int num_of_nodes = WU.size();
    float weight;

    path.push_back(source);
    visited[source] = 1;
    if(source == target) {
        paths.push_back(path);
        return;
    }
    for(int node = 0; node < num_of_nodes; node++) {
        weight = 0.0;
        if(visited[node] == 0) {
            if(source > node) {
                weight = WU[node][source - node];
            }
            else if(source < node) {
                weight = WU[source][node - source];
            }
            if(weight > 0) {
                path[0] += weight;
                dfs_to_target(node, target, WU, visited, paths, path);
                visited[node] = 0;
                path[0] -= weight;
                path.pop_back();
            }
        }
    }
    return;
}

std::vector<std::vector<float>> paths_dfs(int source, int target, const std::vector<std::vector<float>>& WU)
{
    int num_of_nodes = WU.size();
    std::vector<float> path;
    std::vector<std::vector<float>> paths;
    std::vector<short> visited (num_of_nodes, 0);

    path.push_back(0.0);
    dfs_to_target(source, target, WU, visited, paths, path);
    return paths;
}

std::vector<std::vector<float>> cpp_all_paths(const std::vector<std::vector<float>>& WU, const std::vector<int>& targets)
{
    int target = targets[0];
    std::vector<std::vector<float>> paths;
    paths = paths_dfs(0, target, WU);
    // for(int source = 0; source < num_of_nodes; source++) {
    //     if(source != target) {
    //         paths = paths_dfs(source, target, WU);
    //     }
    // }
    //
    // for(auto row: WU) {
    //     for(auto w:row) {
    //         std::cout << w << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // for(auto t: targets) {
    //     std::cout << t << " ";
    // }

    return paths;
}
