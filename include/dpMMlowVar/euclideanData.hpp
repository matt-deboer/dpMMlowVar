/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>

#include <jsCore/clData.hpp>

namespace dplv {

template<typename T>
struct Euclidean //: public DataSpace<T>
{

  class Cluster
  {
    protected:
    Eigen::Matrix<T,Eigen::Dynamic,1> centroid_;
    Eigen::Matrix<T,Eigen::Dynamic,1> xSum_;
    uint32_t N_;

    public:

    Cluster() : centroid_(0,1), xSum_(0,1), N_(0)
    {};

    Cluster(uint32_t D) : centroid_(D,1), xSum_(0,1), N_(0)
    {};

    Cluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i) : centroid_(x_i), xSum_(x_i), N_(1)
    {};

    Cluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& xSum, uint32_t N) :
      centroid_(xSum), xSum_(xSum), N_(N)
    {if(N>0) centroid_/=N_;};

    T dist (const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i) const
    { return Euclidean::dist(this->centroid_, x_i); };

    void computeSS(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x,  const VectorXu& z,
        const uint32_t k)
    {
      const uint32_t D = x.rows();
      const uint32_t N = x.cols();
      N_ = 0;
      xSum_.setZero(D);
      for(uint32_t i=0; i<N; ++i)
        if(z(i) == k)
        {
          xSum_ += x.col(i); 
          ++ N_;
        }
      //TODO: cloud try to do sth more random here
      if(N_ == 0)
        xSum_ = x.col(k); //Matrix<T,Eigen::Dynamic,1>::Zero(D,1);
    };

    void updateCenter()
    {
      assert(this->centroid()(0) == this->centroid()(0));
      if(N_ > 0)
        centroid_ = xSum_/N_;
//      else
//        centroid_ = xSum_;
    };

    void updateSS(const boost::shared_ptr<jsc::ClData<T> >& cld, uint32_t k)
    {
      xSum_ = cld->xSum(k);
      N_ = cld->count(k);
    };

    void updateCenter(const boost::shared_ptr<jsc::ClData<T> >& cld, uint32_t k)
    {
      updateSS(cld,k); 
      updateCenter();
//      cout<<centroid_.transpose()<<endl;
    };

    void resetCenter(const boost::shared_ptr<jsc::ClData<T> >& cld)
    {
      int rid = int(floor(cld->N()*double(std::rand())/double(RAND_MAX)));
      centroid_ = cld->x()->col(rid);
    }

    void computeCenter(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x,  const VectorXu& z,
        const uint32_t k)
    {
      computeSS(x,z,k);
      updateCenter();
    };

    bool isInstantiated() const {return this->N_>0;};

    uint32_t N() const {return N_;};
    uint32_t& N(){return N_;};
    const Eigen::Matrix<T,Eigen::Dynamic,1>& centroid() const {return centroid_;};
     Eigen::Matrix<T,Eigen::Dynamic,1>& centroid() {return centroid_;};
    const Eigen::Matrix<T,Eigen::Dynamic,1>& xSum() const {return xSum_;};
  };

  class DependentCluster : public Cluster
  {
    protected:
    // variables
    T t_;
    T w_;
    // parameters
    T tau_;
    T lambda_;
    T Q_;
    Eigen::Matrix<T,Eigen::Dynamic,1> prevCentroid_;

    public:

    DependentCluster() : Cluster(), t_(0), w_(0), tau_(1), lambda_(1), Q_(1),
      prevCentroid_(this->centroid_)
    {};

    DependentCluster(uint32_t D) : Cluster(D), t_(0), w_(0), tau_(1),
      lambda_(1), Q_(1), prevCentroid_(this->centroid_)
    {};

    DependentCluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i) : Cluster(x_i), t_(0),
      w_(0), tau_(1), lambda_(1), Q_(1), prevCentroid_(this->centroid_)
    {};

    DependentCluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i, T tau, T lambda, T Q) :
      Cluster(x_i), t_(0), w_(0), tau_(tau), lambda_(lambda), Q_(Q),
      prevCentroid_(this->centroid_)
    {};

    DependentCluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i, const DependentCluster& cl0) :
      Cluster(x_i), t_(0), w_(0), tau_(cl0.tau()), lambda_(cl0.lambda()),
      Q_(cl0.Q()), prevCentroid_(this->centroid_)
    {};

    DependentCluster(T tau, T lambda, T Q) :
      Cluster(), t_(0), w_(0), tau_(tau), lambda_(lambda), Q_(Q), 
      prevCentroid_(this->centroid_)
    {};

    DependentCluster(const DependentCluster& b) :
      Cluster(b.xSum(), b.N()), t_(b.t()), w_(b.w()), tau_(b.tau()),
      lambda_(b.lambda()), Q_(b.Q()), prevCentroid_(b.prevCentroid())
    {this->centroid_ = b.centroid();};

    bool isDead() const {return t_*Q_ > lambda_;};
    bool isNew() const {return t_ == 0;};

    void incAge() { ++ t_; };

    const Eigen::Matrix<T,Eigen::Dynamic,1>& prevCentroid() const {return prevCentroid_;};
    Eigen::Matrix<T,Eigen::Dynamic,1>& prevCentroid() {return prevCentroid_;};

    void nextTimeStep()
    {
      this->N_ = 0;
      this->prevCentroid_ = this->centroid_;
    };

    void updateWeight()
    {
      w_ = w_ == 0? this->N_ : 1./(1./w_ + t_*tau_) + this->N_;
      t_ = 0;
    };

    void print() const 
    {
      cout<<"cluster globId="<<globalId
        <<"\tN="<<this->N_ <<"\tage="<<t_ <<"\tweight="
        <<w_ <<"\t dead? "<<this->isDead()
        <<"  center: "<<this->centroid().transpose()<<endl
        <<"  xSum: "<<this->xSum_.transpose()<<endl;
      assert(this->centroid()(0) == this->centroid()(0));
      assert(!(this->centroid().array() == 0).all());
    };

    DependentCluster* clone(){return new DependentCluster(*this);}

    void reInstantiate()
    { const T gamma = 1.0/(1.0/w_ + t_*tau_);
      this->centroid_ = (this->centroid_ * gamma + this->xSum_)/(gamma+this->N_);
    };

    void reInstantiate(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x_i)
    {
      this->xSum_ = x_i; this->N_ = 1;
      reInstantiate();
    };

    T maxDist() const { return this->lambda_;};
    T dist (const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i) const
    {
      if(this->isInstantiated())
        return Euclidean::dist(this->centroid_, x_i);
      else{
        return Euclidean::dist(this->centroid_,x_i) / (tau_*t_+1.+ 1.0/ w_) + Q_*t_; 
      }
    };

    T tau() const {return tau_;};
    T lambda() const {return lambda_;};
    T Q() const {return Q_;};
    T t() const {return t_;};
    T w() const {return w_;};

    uint32_t globalId; // id globally - only increasing id
  };
   
  static T dist(const Eigen::Matrix<T,Eigen::Dynamic,1>& a, const Eigen::Matrix<T,Eigen::Dynamic,1>& b)
  { return (a-b).squaredNorm(); };

  static T dissimilarity(const Eigen::Matrix<T,Eigen::Dynamic,1>& a, const Eigen::Matrix<T,Eigen::Dynamic,1>& b)
  { return (a-b).squaredNorm();};

  static bool closer(const T a, const T b) { return a<b; };

  template<int D>
  static void computeCenters(const std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& xs,
      const std::vector<uint32_t> zs, uint32_t K,
      std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& mus);

  static Eigen::Matrix<T,Eigen::Dynamic,1> computeSum(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, 
      const VectorXu& z, const uint32_t k, uint32_t* N_k);

  static Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> computeCenters(const
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, const VectorXu& z, const uint32_t K, 
      VectorXu& Ns);

  static Eigen::Matrix<T,Eigen::Dynamic,1> computeCenter(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, 
      const VectorXu& z, const uint32_t k, uint32_t* N_k);

  // TODO deprecate soon and use cluster classes instead
  static T distToUninstantiated(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i, const
      Eigen::Matrix<T,Eigen::Dynamic,1>& ps_k, const T t_k, const T w_k, const T tau, 
      const T Q)
  { return dist(x_i, ps_k) / (tau*t_k+1.+ 1.0/w_k) + Q*t_k; };

  static bool clusterIsDead(const T t_k, const T lambda, const T Q)
  { return t_k*Q > lambda;};

  static Eigen::Matrix<T,Eigen::Dynamic,1> reInstantiatedOldCluster(const
      Eigen::Matrix<T,Eigen::Dynamic,1>& xSum, const T N_k, const Eigen::Matrix<T,Eigen::Dynamic,1>& ps_k, const T t_k,
      const T w_k, const T tau);
  
  static T updateWeight(const Eigen::Matrix<T,Eigen::Dynamic,1>& xSum, const uint32_t N_k,
      const Eigen::Matrix<T,Eigen::Dynamic,1>& ps_k, const T t_k, const T w_k, const T tau)
  {return 1./(1./w_k + t_k*tau) + N_k;};

};

//============================= impl ==========================================

template<typename T> template<int D>
void Euclidean<T>::computeCenters(const
    std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& xs, const std::vector<uint32_t>
    zs, uint32_t K, std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& mus) {
  
  for(uint32_t k=0; k<K; ++k) mus[k].fill(0);
  std::vector<uint32_t> Ns(K,0);
  for(uint32_t i=0; i<xs.size(); ++i) {
    mus[zs[i]] += xs[i];
    ++Ns[zs[i]]; 
  }
  // Spherical mean computation
  for(uint32_t k=0; k<K; ++k)
    mus[k] /= Ns[k];
};

  template<typename T>                                                            
Eigen::Matrix<T,Eigen::Dynamic,1> Euclidean<T>::computeSum(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, 
    const VectorXu& z, const uint32_t k, uint32_t* N_k)
{
  const uint32_t D = x.rows();
  const uint32_t N = x.cols();
  Eigen::Matrix<T,Eigen::Dynamic,1> xSum(D);
  xSum.setZero(D);
  if(N_k) *N_k = 0;
  for(uint32_t i=0; i<N; ++i)
    if(z(i) == k)
    {
      xSum += x.col(i); 
      if(N_k) (*N_k) ++;
    }
  return xSum;
};

template<typename T>                                                            
Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> Euclidean<T>::computeCenters(const
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, const VectorXu& z, const uint32_t K, 
    VectorXu& Ns)
{
  const uint32_t D = x.rows();
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> centroids(D,K);
#pragma omp parallel for 
  for(uint32_t k=0; k<K; ++k)
    centroids.col(k) = computeCenter(x,z,k,&Ns(k));
  return centroids;
};

template<typename T>                                                            
Eigen::Matrix<T,Eigen::Dynamic,1> Euclidean<T>::computeCenter(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, 
    const VectorXu& z, const uint32_t k, uint32_t* N_k)
{
  const uint32_t D = x.rows();
  const uint32_t N = x.cols();
  if(N_k) *N_k = 0;
  Eigen::Matrix<T,Eigen::Dynamic,1> mean_k(D);
  mean_k.setZero(D);
  for(uint32_t i=0; i<N; ++i)
    if(z(i) == k)
    {
      mean_k += x.col(i); 
      if(N_k) (*N_k) ++;
    }
  if(!N_k)
    return mean_k; // TODO anoygin
  cout<<mean_k.transpose()<<" N="<<(*N_k)<<endl;
  if(*N_k > 0)
  {
    cout<<"centroid inside "<<endl<<(mean_k/(*N_k))<<endl;
    return mean_k/(*N_k);
  }
  else
    //TODO: cloud try to do sth more random here
    return x.col(k); //Matrix<T,Eigen::Dynamic,1>::Zero(D,1);
};

template<typename T>                                                            
Eigen::Matrix<T,Eigen::Dynamic,1> Euclidean<T>::reInstantiatedOldCluster(const
    Eigen::Matrix<T,Eigen::Dynamic,1>& xSum, const T N_k, const Eigen::Matrix<T,Eigen::Dynamic,1>& ps_k, const T t_k, const
    T w_k, const T tau)
{
  const T gamma = 1.0/(1.0/w_k + t_k*tau);
  return (ps_k*gamma + xSum)/(gamma+N_k);
};

}
