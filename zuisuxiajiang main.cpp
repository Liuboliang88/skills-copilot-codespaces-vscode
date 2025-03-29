// File: src/main.cpp
#include <iostream>
#include <vector>
#include <cstdlib>  // system()
#include "optimizer.hpp"
#include "gjf_generator.hpp"
#include "gaussian_parser.hpp"

int main() {
    // 初始结构（以水分子为例）
    std::vector<Atom> atoms = {
        {"O", 0.0, 0.0, 0.132576},
        {"H", 0.0, 0.733729, -0.530303},
        {"H", 0.0, -0.733729, -0.530303}
    };

    int iteration = 0;
    double alpha = 0.1;               // 最速下降步长（需根据实际情况调试）
    const double conv_threshold = 1e-6;
    const int max_iter = 1000;

    // 用于保存最后一次的梯度
    std::vector<Gradient> finalGrads;

    while (iteration < max_iter) {
        std::cout << "Iteration " << iteration << std::endl;
        
        // 生成 Gaussian 输入文件
        std::string gjf_filename = "opt_step" + std::to_string(iteration) + ".gjf";
        writeGJF(gjf_filename, atoms, iteration);
        
        // 调用 Gaussian 进行计算，假设 Gaussian 命令为 "g16"
        std::string command = "g16 " + gjf_filename;
        int ret = std::system(command.c_str());
        if (ret != 0) {
            std::cerr << "Gaussian calculation failed at iteration " << iteration << std::endl;
            break;
        }
        
        // 解析 Gaussian 输出文件
        // 假设 Gaussian 输出文件为 "opt_step{iteration}.log"
        std::string log_filename = "opt_step" + std::to_string(iteration) + ".log";
        std::vector<Gradient> grads = parse_gradient(log_filename);
        if (grads.empty()) {
            std::cerr << "Failed to parse gradients at iteration " << iteration << std::endl;
            break;
        }
        
        // 保存最后一次梯度
        finalGrads = grads;

        // 检查收敛性
        if (converged(grads, conv_threshold)) {
            std::cout << "Convergence reached at iteration " << iteration << std::endl;
            break;
        }
        
        // 更新原子坐标
        update_coords(atoms, grads, alpha);
        iteration++;
    }
    
    // 输出最终优化结构
    std::cout << "\nOptimized structure:\n";
    for (const auto& atom : atoms) {
        std::cout << atom.symbol << "    "
                  << atom.x << "    "
                  << atom.y << "    "
                  << atom.z << "\n";
    }

    // 输出最终受力（最后一次解析到的梯度）
    std::cout << "\nFinal forces (Hartrees/Bohr):\n";
    for (size_t i = 0; i < finalGrads.size(); ++i) {
        std::cout << "Atom " << (i + 1) << ": "
                  << finalGrads[i].fx << "  "
                  << finalGrads[i].fy << "  "
                  << finalGrads[i].fz << "\n";
    }

    return 0;
}
