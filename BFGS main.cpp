BFGS main.cpp


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include "matrix1.hpp"
#include "optimizer.hpp"
#include "gjf_generator.hpp"
#include "gaussian_parser.hpp"


// ✅ 插入 readAtomsFromGJF 函数实现
std::vector<Atom> readAtomsFromGJF(const std::string& filename) {
    std::ifstream fin(filename);
    std::string line;
    std::vector<Atom> atoms;
    bool inCoords = false;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        if (!inCoords) {
            if (line.find("0") != std::string::npos && line.find("1") != std::string::npos) {
                inCoords = true;
            }
            continue;
        }

        std::string symbol;
        double x, y, z;
        iss >> symbol >> x >> y >> z;
        if (iss.fail() || symbol.empty() || !(iss >> std::ws).eof()) break;

        atoms.push_back({symbol, x, y, z});
    }

    return atoms;
}

// ✅ 主程序入口
int main() {
    std::string initFile = "initial.gjf";
    std::vector<Atom> atoms = readAtomsFromGJF(initFile);
    if (atoms.empty()) {
        std::cerr << "❌ No atoms read from " << initFile << "\n";
        return 1;
    }

    writeGJF("opt_step0.gjf", atoms, 0, false);
    std::system("g16 opt_step0.gjf");

    double E_k = parse_energy("opt_step0.log");
    std::vector<Gradient> grad_k_vec = parse_gradient("opt_step0.log");

    size_t n = atoms.size();
    size_t dim = 3 * n;

    Matrix H(dim, dim, 0.0);
    for (size_t i = 0; i < dim; ++i) H(i, i) = 1.0;

    double alpha = 0.1;
    double threshold = 5e-4;
    int max_iter = 100;

    Matrix x_k(dim, 1);
    Matrix g_k(dim, 1);

    for (size_t i = 0; i < n; ++i) {
        x_k(3 * i + 0) = atoms[i].x;
        x_k(3 * i + 1) = atoms[i].y;
        x_k(3 * i + 2) = atoms[i].z;
    }

    for (size_t i = 0; i < n; ++i) {
        g_k(3 * i + 0) = grad_k_vec[i].fx;
        g_k(3 * i + 1) = grad_k_vec[i].fy;
        g_k(3 * i + 2) = grad_k_vec[i].fz;
    }

    for (int iter = 1; iter <= max_iter; ++iter) {
        // Matrix p = -1.0 * (H % g_k);
        Matrix p = - H.inver() % g_k;
        Matrix x_k1 = x_k + alpha * p;

        for (size_t i = 0; i < n; ++i) {
            atoms[i].x = x_k1(3 * i + 0);
            atoms[i].y = x_k1(3 * i + 1);
            atoms[i].z = x_k1(3 * i + 2);
        }

        writeGJF("opt_step.gjf", atoms, iter, false);
        std::system("g16 opt_step.gjf");

        double E_k1 = parse_energy("opt_step.log");
        std::vector<Gradient> grad_k1_vec = parse_gradient("opt_step.log");

        Matrix g_k1(dim, 1);
        for (size_t i = 0; i < n; ++i) {
            g_k1(3 * i + 0) = grad_k1_vec[i].fx;
            g_k1(3 * i + 1) = grad_k1_vec[i].fy;
            g_k1(3 * i + 2) = grad_k1_vec[i].fz;
        }

        Matrix s = x_k1 - x_k;
        Matrix y = g_k1 - g_k;

        std::cout << "🔁 Iter " << iter << ": Energy = " << E_k1 << std::endl;

        if (grad_k1_vec.empty()) {
            std::cerr << "❌ Gradient extraction failed. Abort.\n";
            break;
        }

        if (converged(grad_k1_vec, threshold)) {
            std::cout << "✅ Converged at iter " << iter << std::endl;
            break;
        }

        bfgs_update(H, s, y);

        x_k = x_k1;
        g_k = g_k1;
        E_k = E_k1;
    }

    std::cout << "\nFinal structure:\n";
    for (const auto& atom : atoms) {
        std::cout << atom.symbol << "\t" << atom.x << "\t" << atom.y << "\t" << atom.z << "\n";
    }

    std::cout << "\nFinal forces (Hartrees/Bohr):\n";
    for (const auto& g : grad_k_vec) {
        std::cout << g.fx << "\t" << g.fy << "\t" << g.fz << "\n";
    }

    return 0;
}
