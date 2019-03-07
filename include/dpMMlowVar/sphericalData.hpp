/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>
#include <jsCore/clData.hpp>

using std::min;
using std::max;

namespace dplv {

/* rotation from point A to B; percentage specifies how far the rotation will 
 * bring us towards B [0,1] */
template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> rotationFromAtoB(const Eigen::Matrix<T,Eigen::Dynamic,1>& a,
    const Eigen::Matrix<T,Eigen::Dynamic,1>& b, T percentage=1.0)
{
  assert(b.size() == a.size());

  uint32_t D_ = b.size();
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> bRa(D_,D_);
   
  T dot = b.transpose()*a;
  ASSERT(fabs(dot) <=1.0, "a="<<a.transpose()<<" |.| "<<a.norm()
      <<" b="<<b.transpose()<<" |.| "<<b.norm()
      <<" -> "<<dot);
  dot = max(static_cast<T>(-1.0),min(static_cast<T>(1.0),dot));
//  cout << "dot="<<dot<<" | |"<<fabs(dot+1.)<<endl;
  if(fabs(dot -1.) < 1e-6)
  {
    // points are almost the same -> just put identity
    bRa =  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Identity(D_,D_);
  }else if(fabs(dot +1.) <1e-6) 
  {
    // direction does not matter since points are on opposing sides of sphere
    // -> pick one and rotate by percentage;
    bRa = -Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Identity(D_,D_);
    bRa(0,0) = cos(percentage*M_PI*0.5);
    bRa(1,1) = cos(percentage*M_PI*0.5);
    bRa(0,1) = -sin(percentage*M_PI*0.5);
    bRa(1,0) = sin(percentage*M_PI*0.5);
  }else{
    T alpha = acos(dot) * percentage;

    Eigen::Matrix<T,Eigen::Dynamic,1> c(D_);
    c = a - b*dot;
    ASSERT(c.norm() >1e-5, "c="<<c.transpose()<<" |.| "<<c.norm());
    c /= c.norm();
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> A = b*c.transpose() - c*b.transpose();

    bRa = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Identity(D_,D_) + sin(alpha)*A + 
      (cos(alpha)-1.)*(b*b.transpose() + c*c.transpose());
  }
  return bRa;
};


template<typename T>
struct Spherical //: public DataSpace<T>
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
      centroid_(xSum/xSum.norm()), xSum_(xSum), N_(N)
    {};

    T dist (const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i) const
    { return Spherical::dist(this->centroid_, x_i); };

    void computeSS(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x,  const VectorXu& z,
        const uint32_t k)
    {
      Spherical::computeSum(x,z,k,&N_);
      //TODO: cloud try to do sth more random here
      if(N_ == 0)
      {
        const uint32_t D = x.rows();
        xSum_ = Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(D);
        xSum_(0) = 1.;
      }
    };

    void updateCenter()
    {
      if(N_ > 0)
        centroid_ = xSum_/xSum_.norm();
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
    T beta_;
    T lambda_;
    T Q_;
    Eigen::Matrix<T,Eigen::Dynamic,1> prevCentroid_;

    public:

    DependentCluster() : Cluster(), t_(0), w_(0), beta_(1), lambda_(1), Q_(1),
      prevCentroid_(this->centroid_)
    {};

    DependentCluster(uint32_t D) : Cluster(D), t_(0), w_(0), beta_(1),
      lambda_(1), Q_(1), prevCentroid_(this->centroid_)
    {};

    DependentCluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i) : Cluster(x_i), t_(0),
      w_(0), beta_(1), lambda_(1), Q_(1), prevCentroid_(this->centroid_)
    {};

    DependentCluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i, T beta, T lambda, T Q) :
      Cluster(x_i), t_(0), w_(0), beta_(beta), lambda_(lambda), Q_(Q), 
      prevCentroid_(this->centroid_)
    {};

    DependentCluster(const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i, const DependentCluster& cl0) :
      Cluster(x_i), t_(0), w_(0), beta_(cl0.beta()), lambda_(cl0.lambda()),
      Q_(cl0.Q()), prevCentroid_(this->centroid_)
    {};

    DependentCluster(T beta, T lambda, T Q) :
      Cluster(), t_(0), w_(0), beta_(beta), lambda_(lambda), Q_(Q), 
      prevCentroid_(this->centroid_)
    {};

    DependentCluster(const DependentCluster& b) :
      Cluster(b.xSum(), b.N()), t_(b.t()), w_(b.w()), beta_(b.beta()),
      lambda_(b.lambda()), Q_(b.Q()), prevCentroid_(b.prevCentroid())
    {};

    DependentCluster* clone(){return new DependentCluster(*this);}

    bool isDead() const {return t_*Q_ < lambda_;};
    bool isNew() const {return t_ == 0;};

    void incAge() { ++ t_; };

    void print() const 
    {
      cout<<"cluster globId="<<globalId
        <<"\tN="<<this->N_ 
        <<"\tage="<<t_ 
        <<"\tweight="<<w_ 
        <<"\t dead? "<<this->isDead()
        <<"  center: "<<this->centroid().transpose()<<endl;
    };

    const Eigen::Matrix<T,Eigen::Dynamic,1>& prevCentroid() const {return prevCentroid_;};
    Eigen::Matrix<T,Eigen::Dynamic,1>& prevCentroid() {return prevCentroid_;};

    void nextTimeStep()
    {
      this->N_ = 0;
      this->prevCentroid_ = this->centroid_;
    };

    void updateWeight()
    {
      T phi, theta, eta;
      T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
              Spherical::dist(this->xSum_,this->centroid_)/this->xSum_.norm())));
      Spherical::solveProblem2(this->xSum_ , zeta, t_, w_, beta_, phi,theta,eta);
      w_ = w_ == 0.0? this->xSum_.norm() : w_ * cos(theta) + beta_*t_*cos(phi)
        + this->xSum_.norm()*cos(eta);
      t_ =  0;
    };

    void reInstantiate()
    {
      T phi, theta, eta;
      T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
              Spherical::dist(this->xSum_,this->prevCentroid_)/this->xSum_.norm())));
      Spherical::solveProblem2(this->xSum_ , zeta, t_, w_, beta_, phi,theta,eta);

      // rotate point from mean_k towards previous mean by angle eta?
      this->centroid_ = rotationFromAtoB<T>(this->xSum_/this->xSum_.norm(), 
          this->prevCentroid_, eta/(phi*t_+theta+eta)) * this->xSum_/this->xSum_.norm(); 
    };

    void reInstantiate(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x_i)
    {
      this->xSum_ = x_i; this->N_ = 1;
      reInstantiate();
    };

    T maxDist() const { return this->lambda_+1.;};
    T dist (const Eigen::Matrix<T,Eigen::Dynamic,1>& x_i) const
    {
      if(this->isInstantiated())
        return Spherical::dist(this->centroid_, x_i);
      else{
        T phi, theta, eta;
        T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
                Spherical::dist(x_i,this->prevCentroid_) )));
//        cout<<"zeta="<<zeta;
        // apprixmation here for small angles -> same as on GPU
        Spherical::solveProblem2Approx(x_i, zeta, t_, w_, beta_, phi,theta,eta);
//        cout<<" phi="<<phi<<" theta="<<theta<<" eta="<<eta<<" zeta="<<zeta
//          <<" w_="<<w_
//          <<" beta="<<beta_<<" Q="<<Q_<<" t="<<t_<<endl;

        return w_*(cos(theta)-1.) + t_*beta_*(cos(phi)-1.) + Q_*t_
          + cos(eta); // no minus 1 here cancels with Z(beta) from the two other assignments
      }
    };

    T beta() const {return beta_;};
    T lambda() const {return lambda_;};
    T Q() const {return Q_;};
    T t() const {return t_;};
    T w() const {return w_;};

    uint32_t globalId; // id globally - only increasing id
  };

  static T dist(const Eigen::Matrix<T,Eigen::Dynamic,1>& a, 
      const Eigen::Matrix<T,Eigen::Dynamic,1>& b)
  { return a.transpose()*b; };

  static T dissimilarity(const Eigen::Matrix<T,Eigen::Dynamic,1>& a, 
      const Eigen::Matrix<T,Eigen::Dynamic,1>& b)
  { return acos(min(static_cast<T>(1.0),max(static_cast<T>(-1.0),
          (a.transpose()*b)(0)))); };

  static bool closer(const T a, const T b)
  { return a > b; };

  template<int D>
  static void computeCenters(const std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& xs,
      const std::vector<uint32_t> zs, uint32_t K,
      std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& mus);

  private:

  static void solveProblem1(T gamma, T age, const T beta, T& phi, 
      T& theta); 
  static void solveProblem2(const Eigen::Matrix<T,Eigen::Dynamic,1>& xSum, T zeta, 
      T age, T w, const T beta, T& phi, T& theta, T& eta); 

  static void solveProblem1Approx(T gamma, T age, const T beta, T& phi, 
      T& theta); 
  static void solveProblem2Approx(const Eigen::Matrix<T,Eigen::Dynamic,1>& xSum, 
      T zeta, T age, T w, const T beta, T& phi, T& theta, T& eta); 

  static Eigen::Matrix<T,Eigen::Dynamic,1> computeSum(const 
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, const VectorXu& z, const uint32_t k,
      uint32_t* N_k);

};



// ================================ impl ======================================

template<typename T> template<int D>
void Spherical<T>::computeCenters(const
    std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& xs, const std::vector<uint32_t>
    zs, uint32_t K, std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1> > >& mus) {
  
  for(uint32_t k=0; k<K; ++k) mus[k].fill(0);
  for(uint32_t i=0; i<xs.size(); ++i)
    mus[zs[i]] += xs[i];
  // Spherical mean computation
  for(uint32_t k=0; k<K; ++k)
    mus[k] /= mus[k].norm();
};

  template<typename T>                                                            
Matrix<T,Eigen::Dynamic,1> Spherical<T>::computeSum(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x, 
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


template<class T>
void Spherical<T>::solveProblem1(T gamma, T age, const T beta, T& phi, T& theta)
{
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  gamma = T phi + theta
  // for phi and theta
  phi = 0.0; 

  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = sin(phi);
    T f = - gamma + age*phi + asin(beta*sinPhi);
    // mathematica
    T df = age + (beta*cos(phi))/sqrt(1.-beta*beta*sinPhi*sinPhi); 
    T dPhi = f/df;
    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta*sin(phi));
};


template<class T>
void Spherical<T>::solveProblem2(const Eigen::Matrix<T,Eigen::Dynamic,1>& xSum, T zeta, 
    T age, T w, const T beta, T& phi, T& theta, T& eta)
{
  // solves
  // w sin(theta) = beta sin(phi) = ||xSum||_2 sin(eta) 
  // eta + T phi + theta = zeta = acos(\mu0^T xSum/||xSum||_2)
  phi = 0.0;

  //  cout<<"w="<<w<<" age="<<age<<" zeta="<<zeta<<endl;

  T L2xSum = xSum.norm();
  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = sin(phi);
    T cosPhi = cos(phi);
    T f = - zeta + asin(beta/L2xSum *sinPhi) + age * phi + asin(beta/w *sinPhi);
    T df = age + (beta*cosPhi)/sqrt(L2xSum*L2xSum -
        beta*beta*sinPhi*sinPhi) + (beta*cosPhi)/sqrt(w*w -
        beta*beta*sinPhi*sinPhi); 

    T dPhi = f/df;

    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<"f="<<f<<" df="<<df<<" phi="<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta/w *sin(phi));
  eta = asin(beta/L2xSum *sin(phi));
};


template<class T>
void Spherical<T>::solveProblem1Approx(T gamma, T age, const T beta, T& phi, T& theta)
{
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  gamma = T phi + theta
  // for phi and theta
  phi = 0.0; 

  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = phi;
    T cosPhi = 1.;
    T f = - gamma + age*phi + asin(beta*sinPhi);
    // mathematica
    T df = age + (beta*cosPhi)/sqrt(1.-beta*beta*sinPhi*sinPhi); 
    T dPhi = f/df;
    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta*sin(phi));
};


template<class T>
void Spherical<T>::solveProblem2Approx(const Eigen::Matrix<T,Eigen::Dynamic,1>& xSum,
    T zeta, T age, T w, const T beta, T& phi, T& theta, T& eta)
{
  // solves
  // w sin(theta) = beta sin(phi) = ||xSum||_2 sin(eta) 
  // eta + T phi + theta = zeta = acos(\mu0^T xSum/||xSum||_2)
  
  phi = zeta/ (beta*(1.+1./w) + age);
  theta = zeta/( 1.+ w*(1. + age/beta) );
  eta = zeta/(1. + 1./w + age/beta);

};

}

namespace jsc {

template<>
double silhouetteClD<double, dplv::Spherical<double> >(const 
    jsc::ClData<double>& cld)
{ 
  if(cld.K()<2) return -1.0;
//  assert(Ns_.sum() == N_);
  Eigen::Matrix<double,Eigen::Dynamic,1> sil(cld.N());
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> xSums = cld.xSums();
#pragma omp parallel for
  for(uint32_t i=0; i<cld.N(); ++i)
  {
    Eigen::Matrix<double,Eigen::Dynamic,1> b = Eigen::Matrix<double,Eigen::Dynamic,1>::Zero(cld.K());
    for(uint32_t k=0; k<cld.K(); ++k)
      if (k == cld.z(i))
        b(k) = 1.-(cld.x()->col(i).transpose()*(xSums.col(k) - cld.x()->col(i)))(0)/static_cast<double>(cld.count(k));
      else
        b(k) = 1.-(cld.x()->col(i).transpose()*xSums.col(k))(0)/static_cast<double>(cld.count(k));
    double a_i = b(cld.z(i)); // average dist to own cluster
    double b_i = cld.z(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<cld.K(); ++k)
      if(k != cld.z(i) && b(k) == b(k) && b(k) < b_i && cld.count(k) > 0)
      {
        b_i = b(k);
      }
    if(a_i < b_i)
      sil(i) = 1.- a_i/b_i;
    else if(a_i > b_i)
      sil(i) = b_i/a_i - 1.;
    else
      sil(i) = 0.;
  }
  return sil.sum()/static_cast<double>(cld.N());
};

template<>
float silhouetteClD<float, dplv::Spherical<float> >(const 
    jsc::ClData<float>& cld)
{ 
  if(cld.K()<2) return -1.0;
//  assert(Ns_.sum() == N_);
  Eigen::Matrix<float,Eigen::Dynamic,1> sil(cld.N());
  Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> xSums = cld.xSums();
#pragma omp parallel for
  for(uint32_t i=0; i<cld.N(); ++i)
  {
    Eigen::Matrix<float,Eigen::Dynamic,1> b = Eigen::Matrix<float,Eigen::Dynamic,1>::Zero(cld.K());
    for(uint32_t k=0; k<cld.K(); ++k)
      if (k == cld.z(i))
        b(k) = 1.-(cld.x()->col(i).transpose()*(xSums.col(k) - cld.x()->col(i)))(0)/static_cast<float>(cld.count(k));
      else
        b(k) = 1.-(cld.x()->col(i).transpose()*xSums.col(k))(0)/static_cast<float>(cld.count(k));
    float a_i = b(cld.z(i)); // average dist to own cluster
    float b_i = cld.z(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<cld.K(); ++k)
      if(k != cld.z(i) && b(k) == b(k) && b(k) < b_i && cld.count(k) > 0)
      {
        b_i = b(k);
      }
    if(a_i < b_i)
      sil(i) = 1.- a_i/b_i;
    else if(a_i > b_i)
      sil(i) = b_i/a_i - 1.;
    else
      sil(i) = 0.;
  }
  return sil.sum()/static_cast<float>(cld.N());
};

}
