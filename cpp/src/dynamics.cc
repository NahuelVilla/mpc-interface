///
/// Gecko - Tools
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
///
/// Author2: David Bellis
/// Copyright2: Nimble One
/// Date2: Aug 2023
///

#include <iostream>
#include <mpc-interface/dynamics.hh>
#include <mpc-interface/tools.hh>

namespace gecko {
namespace tools {

using namespace Eigen;

/*
TODO: DomainVariable and ExtendedSystem should be more compatible, Maybe implementations of
      the same abstract class.
TODO: It should be much easier to access the inputs and states with their sizes, and IDs.
      maybe add an input_ID (as state_ID)
*/

DomainVariable::DomainVariable(){
}

void DomainVariable::init(
  std::vector<std::string> & names,
  std::vector<int> & sizes,
  std::vector<std::string> & axes,
  bool time_variant,
  std::function<void(
    void ** objects,
    std::map<std::string, double> & kargs
  )> how_to_update_size
) {
  // TODO: names and sizes must have the same amount of elements

  time_variant_ = time_variant;
  how_to_update_size_ = how_to_update_size;
  axes_ = axes;

  // more complex construction
  identify_domain(names);
  set_sizes(names, sizes);

  // outputs is std::vector
  // definitions is std::map
  make_definitions();
}

void DomainVariable::__figuring_out(
  void ** objects,
  std::map<std::string, double> & kargs
){
    if(how_to_update_size_ && time_variant_)
        how_to_update_size_(objects, kargs);
}

void DomainVariable::identify_domain(std::vector<std::string> & names) {
  domain_ID.clear();
  for(const auto & axis : axes_) // iterate across vector
  {
    for(size_t i = 0; i < names.size(); i++) // loop across vector so we have index
      domain_ID.insert({ names[i] + axis, i }); // insert into map
  }
}

void DomainVariable::set_sizes(std::vector<std::string> & names, std::vector<int> & sizes) {
  (void)names;
  domain.clear();
  for(const auto & dID: domain_ID) // iterate across map
  {
    domain.insert({dID.first, sizes[dID.second]}); // insert into other map
  }
  all_variables.clear();
  for(const auto & d: domain) // iterate across map
  {
    all_variables.insert({d.first, d.second}); // insert into other map
  }
}

void DomainVariable::update_sizes(std::map<std::string, double> & kargs) {
    void *objects[] = { static_cast<void*>(this)};
    __figuring_out(objects, kargs);
}

void DomainVariable::make_definitions() {
    for(const auto & d: domain) // iterate across map
    {
        std::map<std::string, Eigen::MatrixXd> combination;
        combination.insert({ d.first, MatrixXd::Identity(d.second, d.second)});
        //
        definitions.insert(
          { d.first,
            LineCombo(combination, nullptr, false,
              gecko::tools::do_not_update)
          }
        );
        definitions[d.first]._coefficients.push_back("I");
    }
}

void DomainVariable::define_output(
  std::string & name,
  std::map<std::string, int> & combination,
  bool time_variant,
  std::function<void(
    void ** objects,
    std::map<std::string, double> & kargs
  )> how_to_update) {
    //This function incorporates additional (output) definitions related to
    //the extended system.
    //If needed, the function "how_to_update" can only have one karg called
    //"domVar" which refers to the extended system where the output is
    //defined.

    for(const auto & axis : axes_) // iterate over vector
    {
        for(const auto & c : combination)
        {
            std::map<std::string, Eigen::MatrixXd> combination;
            combination.insert({c.first + axis, MatrixXd::Identity(c.second, c.second)});
            //
            definitions.insert(
              { name + axis,
                LineCombo(combination, nullptr, time_variant, how_to_update)
              }
            );
        }
        outputs.push_back(name + axis);
    }
}

void DomainVariable::update_definitions() {
  for(const auto & d : domain) // iterate over map
  {
    definitions[d.first].matrices[0] =  MatrixXd::Identity(d.second, d.second);
  }

  for(const auto & output : outputs) // iterate over vector
  {
    std::map<std::string, double> kargs;
    kargs.insert({"domVar", -1});
    definitions[output].update(kargs);
  }
}

void DomainVariable::update(std::map<std::string, double> & kargs) {
  update_sizes(kargs);
  update_definitions();
}

ControlSystem::ControlSystem(
    Eigen::MatrixXd *A, Eigen::MatrixXd *B
) : A_(A->rows(), A->cols()), B_(B->rows(), B->cols()){
}

void ControlSystem::init(
  std::vector<std::string> &input_names,
  std::vector<std::string> &state_names,
  Eigen::MatrixXd *A, Eigen::MatrixXd *B,
  std::vector<std::string> &axes,
  bool time_variant,
  std::function<void(
    void ** objects,
    std::map<std::string, double> & kargs
  )> how_to_update_matrices
) {
  input_names_ = input_names;
  state_names_ = state_names;
  for(int i = 0; i < A->rows(); i++)
  {
    for(int j = 0; j < A->cols(); j++)
    {
      A_(i, j) = (*A)(i,j);
    }
  }
  for(int i = 0; i < B->rows(); i++)
  {
    for(int j = 0; j < B->cols(); j++)
    {
      B_(i, j) = (*B)(i,j);
    }
  }
  axes_ = axes;
  time_variant_ = time_variant;
  how_to_update_matrices_ = how_to_update_matrices;
}

void ControlSystem::update_matrices(
  void ** objects,
  std::map<std::string, double> & kargs) {
  how_to_update_matrices_(objects, kargs);
}

/// getters
std::vector<std::string> & ControlSystem::get_input_names()
{
  return input_names_;
}

std::vector<std::string> & ControlSystem::get_state_names()
{
  return state_names_;
}

Eigen::MatrixXd * ControlSystem::get_matrix_A()
{
  return &A_;
}

Eigen::MatrixXd * ControlSystem::get_matrix_B()
{
  return &B_;
}

std::vector<std::string> & ControlSystem::get_axes()
{
  return axes_;
}

bool ControlSystem::is_time_variant()
{
    return time_variant_;
}

/*
TODO: Make a form to deal with axis names longer (or shorter) than 2 characters or
      rise an error when the axes have more (or less) than 2 characters
TODO: the previous point can be done with variables of hte form tuple(name, axis)
      which is immutable and we can separate name and axis easily.
TODO: Report some how what should be in the **kargs for update functions
*/

void how_to_update_matrices_for_extended_control_system(
  void ** objects,
  std::map<std::string, double> & kargs)
{
  gecko::tools::ControlSystem * ctr_syst =
    static_cast<gecko::tools::ControlSystem*>(objects[1]);
  ctr_syst->update_matrices(objects, kargs);

  gecko::tools::ExtendedSystem * ext_syst =
    static_cast<gecko::tools::ExtendedSystem*>(objects[0]);
  //S is the tensor to multiply state
  //U is the tensor to multiply input
  gecko::tools::extend_matrices(ext_syst->get_S(), ext_syst->get_U(), ext_syst->get_horizon_length(),
    ctr_syst->get_matrix_A(), ctr_syst->get_matrix_B());
}

ExtendedSystem::ExtendedSystem() {
}

void ExtendedSystem::init(
  std::vector<std::string> &input_names,
  std::vector<std::string> &state_names,
  std::string &state_vector_name,
  std::vector<std::string> &axes,
  bool time_variant,
  std::function<void(
    void ** objects,
    std::map<std::string, double> & kargs
  )> how_to_update_matrices
) {
  input_names_ = input_names;
  state_names_ = state_names;
  state_vector_name_ = state_vector_name;
  axes_ = axes;
  horizon_length_ = 0;
  time_variant_ = time_variant;
  how_to_update_matrices_ = how_to_update_matrices;

  identify_domain(input_names_, state_names_);
}

void ExtendedSystem::init_from_control_system(
  ControlSystem * control_system,
  std::string state_vector_name,
  unsigned int horizon_length
) {
  input_names_ = control_system->get_input_names();
  state_names_ = control_system->get_state_names();
  state_vector_name_ = state_vector_name;
  gecko::tools::extend_matrices(S_, U_, horizon_length,
    control_system->get_matrix_A(), control_system->get_matrix_B());
  axes_ = control_system->get_axes();
  horizon_length_ = horizon_length;
  time_variant_ = control_system->is_time_variant();
  if(time_variant_)
    how_to_update_matrices_ = how_to_update_matrices_for_extended_control_system;
  else
    how_to_update_matrices_ =  gecko::tools::do_not_update;
}

void ExtendedSystem::identify_domain(std::vector<std::string> &input_name,
                                     std::vector<std::string> &state_names) {
  /// Build domain ID
  std::map<std::string, int> ldomain_ID;
  for (std::size_t i = 0; i < input_name.size(); i++)
    ldomain_ID[input_name[i]] = static_cast<int>(i);

  std::string state_vec_name_dom(state_vector_name_ + "0");
  ldomain_ID[state_vec_name_dom] = static_cast<int>(input_name.size());

  /// Build state ID
  std::map<std::string, int> lstate_ID;
  for (std::size_t i = 0; i < state_names.size(); i++)
    lstate_ID[state_names[i]] = static_cast<int>(i);

  if (axes_.size() == 0) {
    domain_ID_ = ldomain_ID;
    state_ID_ = lstate_ID;
  } else {
    for (std::size_t axes_ind = 0; axes_ind < axes_.size(); axes_ind++) {
      for (auto domain_ID_it = ldomain_ID.begin();
           domain_ID_it != ldomain_ID.end(); domain_ID_it++)
        domain_ID_[domain_ID_it->first + axis_[axes_ind]] =
            domain_ID_it->second;

      for (auto state_ID_it = lstate_ID.begin(); state_ID_it != lstate_ID.end();
           state_ID_it++)
        state_ID_[state_ID_it->first + axis_[axes_ind]] = state_ID_it->second;
    }
  }
}

/// getters
Eigen::Tensor<double, 3> & ExtendedSystem::get_S()
{
    return S_;
}

Eigen::Tensor<double, 4> & ExtendedSystem::get_U()
{
    return U_;
}

int ExtendedSystem::get_horizon_length()
{
  return horizon_length_;
}

void ExtendedSystem::set_sizes() {
  // Merge the two maps inside the all_variables_one.
  all_variables_.insert(domain_ID_.begin(), domain_ID_.end());
  all_variables_.insert(state_ID_.begin(), state_ID_.end());
}

// TODO
void ExtendedSystem::update_sizes() {
  if (time_variant_) {
  }
}

}  // namespace tools
}  // namespace gecko
