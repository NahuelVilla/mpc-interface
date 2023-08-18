///
/// Gecko - Tools
/// Author: David Bellis
/// Copyright: Nimble One
/// Date: Aug 2023
///

#include <mpc-interface/combinations.hh>

namespace nimbleone {
namespace mpc {

using namespace Eigen;

LineCombo::LineCombo(std::map<std::string, Eigen::MatrixXd> combination)
    : combination_(combination) {

    // more complicated construction
    for (const auto & c : combination_)
    {
        variables.push_back(c.first);
        matrices.push_back(c.second);
    }

    for(size_t i =0; i < variables.size(); i++)
    {
        _coefficients.push_back(std::string("C") + std::to_string(i));
    }
}

MatrixXd & LineCombo::getitem(std::string variable)
{
    return combination_[variable];
}

std::map<std::string, Eigen::MatrixXd> & LineCombo::items()
{
    return combination_;
}

std::vector<std::string> & LineCombo::keys()
{
    return variables;
}

std::vector<Eigen::MatrixXd> & LineCombo::values()
{
    return matrices;
}

std::string LineCombo::output()
{
    std::string output;
    std::vector<std::string>::iterator itr_coeffs = _coefficients.begin();
    std::vector<std::string>::iterator itr_vars = variables.begin();
    while(itr_coeffs != _coefficients.end() && itr_vars != variables.end())
    {
        output += *itr_coeffs;
        output += " ( ";
        output += *itr_vars;
        output += " ) ";

        std::vector<std::string>::iterator itr_coeffs_next = itr_coeffs;
        itr_coeffs_next++;
        if(itr_coeffs_next != _coefficients.end())
        {
            output += " + ";
        }

        itr_coeffs++;
        itr_vars++;
    }

    return output;
}

}  // namespace mpc
}  // namespace nimbleone
