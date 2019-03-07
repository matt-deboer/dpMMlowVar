/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <jsCore/global.hpp>
#include <jsCore/clDataGpu.hpp>
#include <jsCore/gpuMatrix.hpp>

#include <dpMMlowVar/euclideanData.hpp>
#include <dpMMlowVar/sphericalData.hpp>
#include <dpMMlowVar/kmeans.hpp>


using namespace Eigen;
using std::cout;
using std::endl;

void spkmLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z,                      
        uint32_t K, uint32_t N);
void spkmLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z,                      
        uint32_t K, uint32_t N);

void kmeansLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z, uint32_t K,
    uint32_t N);
void kmeansLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, uint32_t K,
    uint32_t N);

namespace dplv {

template<class T, class DS>
class KMeansCUDA : public KMeans<T,DS>
{
public:
  KMeansCUDA(const boost::shared_ptr<jsc::ClDataGpu<T> >& cld);
  virtual ~KMeansCUDA();

  virtual void updateLabels();
  virtual void updateCenters();
  virtual void nextTimeStepGpu(T* d_x, uint32_t N, uint32_t step, uint32_t
      offset, bool reset = false);

  void getZfromGpu() {this->cld_->z();};
  uint32_t* d_z(){ return this->cdl_->d_z();};

protected:

  jsc::GpuMatrix<T> d_p_;
  virtual void setupComputeLabelsGPU();
};
typedef KMeansCUDA<double, Euclidean<double> > kmeansCUDAd;
typedef KMeansCUDA<float, Euclidean<float> > kmeansCUDAf;
typedef KMeansCUDA<double, Spherical<double> > spkmCUDAd;
typedef KMeansCUDA<float, Spherical<float> > spkmCUDAf;

// ------------------------- impl --------------------------------------
template<class T, class DS>
KMeansCUDA<T,DS>::KMeansCUDA( const boost::shared_ptr<jsc::ClDataGpu<T> >& cld)
  : KMeans<T,DS>(cld), d_p_(this->D_,cld->K())
{}

template<class T, class DS>
KMeansCUDA<T,DS>::~KMeansCUDA()
{}

template<class T, class DS>
void KMeansCUDA<T,DS>::nextTimeStepGpu(T* d_x, uint32_t N, uint32_t step,
    uint32_t offset, bool reset) 
{ 
  if(reset)
  {
    this->cls_.clear();
    for (uint32_t k=0; k<this->K_; ++k)
      this->cls_.push_back(boost::shared_ptr<typename DS::DependentCluster >(
            new typename DS::DependentCluster()));
  }
  this->cld_->updateData(d_x,N,step,offset);
  this->N_ = this->cld_->N();
  if(reset)
  {
    this->cld_->randomLabels(this->K_);
    this->cld_->updateLabels(this->K_);
    this->cld_->computeSS();
    for(uint32_t k=0; k<this->K_; ++k)
      this->cls_[k]->updateCenter(this->cld_,k);
  }else{
    for (uint32_t k=0; k<this->K_; ++k)
      if(this->cls_[k]->N() == 0)
        this->cls_[k]->resetCenter(this->cld_);
  }
};

template<class T, class DS>
void KMeansCUDA<T,DS>::updateCenters()
{
  this->prevNs_.resize(this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    this->prevNs_(k) = this->cls_[k]->N();

//  this->cld_->updateLabels(this->K_);
  this->cld_->computeSS();
  for(uint32_t k=0; k<this->K_; ++k)
    this->cls_[k]->updateCenter(this->cld_,k);
}

template<class T, class DS>
void KMeansCUDA<T,DS>::setupComputeLabelsGPU()
{
  Matrix<T,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    ps.col(k) = this->cls_[k]->centroid();
  d_p_.set(ps);
};

template<class T, class DS>
void KMeansCUDA<T,DS>::updateLabels()
{
  this->setupComputeLabelsGPU();
  assert(false);
};

template<>
void KMeansCUDA<double,Spherical<double> >::updateLabels()
{
  this->setupComputeLabelsGPU();
  spkmLabels_gpu(this->cld_->d_x(),d_p_.data(),this->cld_->d_z(),
      this->K_, this->cld_->N());
}

template<>
void KMeansCUDA<float,Spherical<float> >::updateLabels()
{
  this->setupComputeLabelsGPU();
  spkmLabels_gpu(this->cld_->d_x(),d_p_.data(),this->cld_->d_z(),
      this->K_, this->cld_->N());
}

template<>
void KMeansCUDA<double,Euclidean<double> >::updateLabels()
{
  this->setupComputeLabelsGPU();
  kmeansLabels_gpu(this->cld_->d_x(),d_p_.data(),this->cld_->d_z(),
      this->K_, this->cld_->N());
}

template<>
void KMeansCUDA<float,Euclidean<float> >::updateLabels()
{
  this->setupComputeLabelsGPU();
  kmeansLabels_gpu(this->cld_->d_x(),d_p_.data(),this->cld_->d_z(),
      this->K_, this->cld_->N());
}

}
