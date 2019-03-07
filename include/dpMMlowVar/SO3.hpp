/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

namespace dplv {

template<typename T>
class SO3
{
  public:
    static inline Eigen::Matrix<T,3,1> vee(const Eigen::Matrix<T,3,3>& W);
    static inline Eigen::Matrix<T,3,3> invVee(const Eigen::Matrix<T,3,1>& w);
    static inline Eigen::Matrix<T,3,3> expMap(const Eigen::Matrix<T,3,1>& w);

    static inline Eigen::Matrix<T,Eigen::Dynamic,1> vee(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& W);
    static inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> invVee(const Eigen::Matrix<T,Eigen::Dynamic,1>& w);
    static inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> logMapW(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& R);
    static inline Eigen::Matrix<T,Eigen::Dynamic,1> logMap(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& R);
    static inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> expMap(const Eigen::Matrix<T,Eigen::Dynamic,1>& w);
    static inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> expMap(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& W);
    static inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> meanRotation(const
        vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& Rs, uint32_t Tmax=100);
    static inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> meanRotation(const
        vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& Rs, Eigen::Matrix<T,Eigen::Dynamic,1> w, uint32_t
        Tmax=100);
};

// ----------------------------- impl ----------------------------------

template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,1> SO3<T>::vee(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& W)
{
  const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> A = 0.5*(W - W.transpose());
  Eigen::Matrix<T,Eigen::Dynamic,1> w(3);
  w << A(2,1), A(0,2), A(1,0);
  return w;
};

template<typename T>
inline Eigen::Matrix<T,3,1> SO3<T>::vee(const Eigen::Matrix<T,3,3>& W)
{
  const Eigen::Matrix<T,3,3> A = 0.5*(W - W.transpose());
  Eigen::Matrix<T,3,1> w(A(2,1), A(0,2), A(1,0));
  return w;
};

template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> SO3<T>::invVee(const Eigen::Matrix<T,Eigen::Dynamic,1>& w)
{
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> W = Eigen::MatrixXf::Zero(3,3);
  W(2,1) = w(0);
  W(0,2) = w(1);
  W(1,0) = w(2);

  W(1,2) = -w(0);
  W(2,0) = -w(1);
  W(0,1) = -w(2);
  return W;
};

template<typename T>
inline Eigen::Matrix<T,3,3> SO3<T>::invVee(const Eigen::Matrix<T,3,1>& w)
{
  Eigen::Matrix<T,3,3> W = Eigen::Matrix<T,3,3>::Zero();
  W(2,1) = w(0);
  W(0,2) = w(1);
  W(1,0) = w(2);
  W(1,2) = -w(0);
  W(2,0) = -w(1);
  W(0,1) = -w(2);
  return W;
};

template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> SO3<T>::logMapW(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& R)
{
  assert(R.rows() ==3);
  assert(R.cols() ==3);
  const T theta = acos((R.trace()-1.)*0.5);
//  cout<<"theta="<<theta<<endl;
  T a = theta/(2.*sin(theta));
  if(a!=a) a = 0.0;
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> W = a*(R-R.transpose());
  return W;
};

template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,1> SO3<T>::logMap(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& R)
{
  return vee(logMapW(R));
};

template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> SO3<T>::expMap(const Eigen::Matrix<T,Eigen::Dynamic,1>& w)
{
  assert(w.rows() == 3);
  const T theta = sqrt(w.array().square().matrix().sum());
//  cout<<"theta="<<theta<<endl;
  const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> W = invVee(w);
  T a = sin(theta)/theta;
  if(a!=a) a = 0.0;
  T b = (1.-cos(theta))/(theta*theta);
  if(b!=b) b = 0.0;
  const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> R = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Identity(3,3) + a * W + b * W*W;
//  cout<<"W"<<endl<<W<<endl;
//  cout<<"W*W"<<endl<<W*W<<endl;
//  cout<<"Rdet="<<R.determinant()<<endl;
  return R;
};

template<typename T>
inline Eigen::Matrix<T,3,3> SO3<T>::expMap(const Eigen::Matrix<T,3,1>& w)
{
  const T theta = sqrt(w.array().square().matrix().sum());
//  cout<<"theta="<<theta<<endl;
  const Eigen::Matrix<T,3,3> W = invVee(w);
  T a = sin(theta)/theta;
  if(a!=a) a = 0.0;
  T b = (1.-cos(theta))/(theta*theta);
  if(b!=b) b = 0.0;
  const Eigen::Matrix<T,3,3> R = Eigen::Matrix<T,3,3>::Identity() + a * W + b * W*W;
//  cout<<"W"<<endl<<W<<endl;
//  cout<<"W*W"<<endl<<W*W<<endl;
//  cout<<"Rdet="<<R.determinant()<<endl;
  return R;
};

template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> SO3<T>::expMap(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& W)
{
  return expMap(invVee(W));
};

/* compute the mean rotation using karcher mean on SO3 */
template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> SO3<T>::meanRotation(const vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& Rs, uint32_t Tmax)
{
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> x(3,Rs.size());
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> muR = Rs[0];// arbitrarily 
  Eigen::Matrix<T,Eigen::Dynamic,1> xMean;
  for(uint32_t t=0; t<Tmax; ++t)
  {
//    muR = 0.5*(muR-muR.transpose()); // symmetrize here 
    for(uint32_t i=0; i<Rs.size(); ++i)
      x.col(i) = logMap(muR.transpose()*Rs[i]);
    xMean = x.rowwise().sum()/x.cols();
    muR = expMap(xMean)*muR;
//    cout<<"@t"<<t<<": "<<xMean.transpose()<<endl;
//    cout<<x<<endl;
    if((xMean.array().abs()<1e-6).all()) break;
  }
  return muR;
}

/* compute the weighted mean rotation using karcher mean on SO3 */
template<typename T>
inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> SO3<T>::meanRotation(const vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >& Rs, Eigen::Matrix<T,Eigen::Dynamic,1> w, uint32_t Tmax)
{
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> x(3,Rs.size());
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> muR = Rs[0];// arbitrarily 
  Eigen::Matrix<T,Eigen::Dynamic,1> xMean;
  for(uint32_t t=0; t<Tmax; ++t)
  {
//    muR = 0.5*(muR-muR.transpose()); // symmetrize here 
    for(uint32_t i=0; i<Rs.size(); ++i)
      x.col(i) = logMap(muR.transpose()*Rs[i])*w(i);
    xMean = x.rowwise().sum()/w.sum();
    muR = expMap(xMean)*muR;
//    cout<<"@t"<<t<<": "<<xMean.transpose()<<endl;
//    cout<<x<<endl;
    if((xMean.array().abs()<1e-6).all()) break;
  }
  return muR;
}

}
