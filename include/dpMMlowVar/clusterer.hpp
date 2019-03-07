/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <fstream>
#include <Eigen/Dense>

#include <jsCore/global.hpp>
#include <jsCore/clData.hpp>

using std::vector;

namespace dplv {

template<class T, class DS>
class Clusterer
{
public:
  Clusterer(const boost::shared_ptr<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& spx, uint32_t K);
  Clusterer(const boost::shared_ptr<jsc::ClData<T> >& cld);
  virtual ~Clusterer();

//  void initialize(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x);

  virtual void updateLabels() = 0;
  virtual void updateCenters() = 0;
  virtual MatrixXu mostLikelyInds(uint32_t n, 
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& deviates) = 0;
  virtual T avgIntraClusterDeviation() = 0;

  virtual void nextTimeStep(const boost::shared_ptr<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& spx);
  virtual void updateState(){}; // after converging for a single time instant
  
  const VectorXu& z() const {return (this->cld_->z());};
  VectorXu counts() const {
    VectorXu Ns(K_);
    for(uint32_t k=0; k<K_; ++k) Ns(k) = cls_[k]->N();
    return Ns;
  };
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> centroids() const {
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> ps(D_,K_);
    for(uint32_t k=0; k<K_; ++k)                                                
      ps.col(k) = cls_[k]->centroid();
    return ps;
  };

  uint32_t globalInd(uint32_t k) const {return cls_[k]->globalId;};

  // natural distance to be used by the algorithm
//  virtual T dist(const Eigen::Matrix<T,Eigen::Dynamic,1>& a, const Eigen::Matrix<T,Eigen::Dynamic,1>& b) = 0;
  // closer in the sense of distance defined above
//  virtual bool closer(T a, T b) = 0;
  // measure of disimilarity between two points (not necessarily the distance)
//  virtual T dissimilarity(const Eigen::Matrix<T,Eigen::Dynamic,1>& a, const Eigen::Matrix<T,Eigen::Dynamic,1>& b) = 0;

  virtual uint32_t getK(){return K_;};
  virtual uint32_t K(){return K_;}; 
  virtual bool converged(T eps=1e-6){
//    cout<<cost_<<" "<<prevCost_<<" "<<fabs(cost_-prevCost_)<<endl ; 
    return fabs(cost_-prevCost_)<eps;}; 
  virtual T cost(){return cost_;}; 

  virtual T silhouette();

  virtual void dumpStats(std::ofstream& fout);

protected:
  uint32_t K_;
  const uint32_t D_;
  uint32_t N_;
  T cost_, prevCost_;
  boost::shared_ptr<jsc::ClData<T> > cld_;
//  boost::shared_ptr<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > spx_; // pointer to data
  vector< boost::shared_ptr<typename DS::DependentCluster> > cls_; // clusters
//  VectorXu z_; // labels
};

// ----------------------------- impl -----------------------------------------
template<class T, class DS>
Clusterer<T,DS>::Clusterer( const boost::shared_ptr<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& spx,
    uint32_t K)
  : K_(K), D_(spx->rows()), N_(spx->cols()), cost_(INFINITY), prevCost_(INFINITY),
  cld_(new jsc::ClData<T>(spx,K))
{
  for (uint32_t k=0; k<K_; ++k)
  {
    cls_.push_back(boost::shared_ptr<typename DS::DependentCluster >(
          new typename DS::DependentCluster(D_)));
    cls_[k]->globalId = k;
  }
};

template<class T, class DS>
Clusterer<T,DS>::Clusterer(const boost::shared_ptr<jsc::ClData<T> >& cld)
  : K_(cld->K()), D_(cld->D()), N_(cld->N()), cost_(INFINITY), prevCost_(INFINITY),
  cld_(cld)
{
  for (uint32_t k=0; k<K_; ++k)
  {
    cls_.push_back(boost::shared_ptr<typename DS::DependentCluster >(
          new typename DS::DependentCluster(D_)));
    cls_[k]->globalId = k;
  }
};

template<class T, class DS>
void Clusterer<T,DS>::nextTimeStep(const boost::shared_ptr<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& spx)
{
  // reset cluster centers
  cls_.clear();
  for (uint32_t k=0; k<K_; ++k)
    cls_.push_back(boost::shared_ptr<typename DS::DependentCluster >(new typename DS::DependentCluster()));
  // update the data
  this->cld_->updateData(spx);
  this->N_ = this->cld_->N();

  this->cld_->randomLabels(K_);
  this->cld_->updateLabels(K_);
  this->cld_->computeSS();
  for(uint32_t k=0; k<this->K_; ++k)
    this->cls_[k]->updateCenter(this->cld_,k);
};

template<class T, class DS>
Clusterer<T,DS>::~Clusterer()
{};

template<class T, class DS>
T Clusterer<T,DS>::silhouette()
{ 
  this->cld_->computeSS();
  return jsc::silhouetteClD<T,DS>(*this->cld_);
//  if(K_<2) return -1.0;
////  assert(Ns_.sum() == N_);
//  Eigen::Matrix<T,Eigen::Dynamic,1> sil(N_);
//#pragma omp parallel for
//  for(uint32_t i=0; i<N_; ++i)
//  {
//    Eigen::Matrix<T,Eigen::Dynamic,1> b = Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(K_);
//    for(uint32_t j=0; j<N_; ++j)
//      if(j != i)
//      {
//        b(cld_->z(j)) += DS::dissimilarity(cld_->x()->col(i),cld_->x()->col(j));
//      }
//    for (uint32_t k=0; k<K_; ++k) b(k) /= cls_[k]->N();
////    b *= Ns_.cast<T>().cwiseInverse(); // Assumes Ns are up to date!
//    T a_i = b(cld_->z(i)); // average dist to own cluster
//    T b_i = cld_->z(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
//    for(uint32_t k=0; k<K_; ++k)
//      if(k != cld_->z(i) && b(k) == b(k) && b(k) < b_i && cls_[k]->N() > 0)
//      {
//        b_i = b(k);
//      }
//    if(a_i < b_i)
//      sil(i) = 1.- a_i/b_i;
//    else if(a_i > b_i)
//      sil(i) = b_i/a_i - 1.;
//    else
//      sil(i) = 0.;
//  }
//  return sil.sum()/static_cast<T>(N_);
};

template<class T, class DS>
void Clusterer<T,DS>::dumpStats(std::ofstream& fout)
{
  fout<<this->K_<<" "<<this->cost_<<" ";
  for(uint32_t k=0; k< this->K_; ++k)
    fout<<this->cls_[k]->N()<<" ";
  for(uint32_t k=0; k< this->K_-1; ++k)
    fout<<(this->cls_[k]->globalId)<<" ";
  fout<<(this->cls_[this->K_-1]->globalId)<<endl;
};
}
