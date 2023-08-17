///
/// Gecko - Tools
/// Author: David Bellis
/// Copyright: Nimble One
/// Date: Aug 2023
///

#include <mpc-interface/combinations.hh>

namespace gecko {
namespace tools {

using namespace Eigen;

LineCombo::LineCombo(
  std::map<std::string, Eigen::MatrixXd> combination,
  std::vector<std::string> * data,
  bool time_variant,
  std::function<void(
    void ** objects,
    std::map<std::string, double> & kargs
  )> how_to_update
) : combination_(combination), data_(data) {

  time_variant_ = time_variant;
  how_to_update_ = how_to_update;

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

void LineCombo::__figuring_out(
  void ** objects,
  std::map<std::string, double> & kargs
){
    if(how_to_update_ && time_variant_)
        how_to_update_(objects, kargs);
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

void LineCombo::update(std::map<std::string, double> & kargs)
{
    void *objects[] =  { static_cast<void*>(this) };
    __figuring_out(objects, kargs);
}

}  // namespace tools
}  // namespace gecko
