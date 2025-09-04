#include <bits/stdc++.h>

#include <armadillo>
#include <ensmallen.hpp>
#include <ensmallen_bits/callbacks/progress_bar.hpp>
#include <ensmallen_bits/function.hpp>
#include <ensmallen_bits/sgd/sgd.hpp>
#include <mlpack.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/tree/octree/octree.hpp>
#include <mlpack/core/tree/statistic.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>

using namespace mlpack;


template <typename VecType = arma::vec>
class CenterOfMassStatistic
{
 public:
  CenterOfMassStatistic() : centerOfMass() {}

  template <typename TreeType = mlpack::Octree<>>
  CenterOfMassStatistic(const TreeType &node)
      : centerOfMass(node.Dataset().n_rows, arma::fill::zeros)
  {
    if (node.IsLeaf())
    {
      for (size_t i = 0; i < node.NumDescendants(); i++)
      {
        centerOfMass += node.Dataset().col(node.Point(i));
      }
      centerOfMass /= node.NumDescendants();
    }
    else
    {
      for (size_t i = 0; i < node.NumChildren(); i++)
      {
        centerOfMass += node.Child(i).NumDescendants() *
                        node.Child(i).Stat().CenterOfMass();
      }
      centerOfMass /= node.NumDescendants();
    }
  }

  const VecType &CenterOfMass() const { return centerOfMass; }

 private:
  VecType centerOfMass;
};

class ExactTSNE;
class DualTreeTSNE;
class BarnesHutTSNE;

template <typename Method,
          typename DistanceType,
          typename TreeType,
          typename MatType = arma::mat,
          typename VecType = arma::vec>
class TSNERules
{
  static_assert(std::is_same_v<Method, DualTreeTSNE> ||
                std::is_same_v<Method, BarnesHutTSNE>);

 public:
  using BoundType = HRectBound<DistanceType>;

  TSNERules(double &sumQ,
            MatType &negF,
            const MatType &embedding,
            const std::vector<size_t> &oldFromNew,
            const double theta = 0.1)
      : sumQ(sumQ), negF(negF), embedding(embedding), oldFromNew(oldFromNew),
        theta(theta)
  {
    // Nothing To Do Here
  }

  double BaseCase(const size_t queryIndex, const size_t referenceIndex)
  {
    if (queryIndex == referenceIndex)
      return 0.0;

    const VecType &queryPoint = embedding.col(oldFromNew[queryIndex]);
    const VecType &referencePoint = embedding.col(oldFromNew[referenceIndex]);
    const double distance = DistanceType::Evaluate(queryPoint, referencePoint);

    const double q = 1.0 / (1.0 + distance);
    sumQ += q;
    negF.col(oldFromNew[queryIndex]) += q * q * (queryPoint - referencePoint);
    if constexpr (std::is_same_v<Method, DualTreeTSNE>)
      negF.col(oldFromNew[referenceIndex]) +=
          q * q * (referencePoint - queryPoint);

    return distance;
  }

  double Score(const size_t queryIndex, TreeType &referenceNode)
  {
    const VecType &queryPoint = embedding.col(oldFromNew[queryIndex]);
    const VecType &referencePoint = referenceNode.Stat().CenterOfMass();
    const double distance = std::max(
        arma::datum::eps, DistanceType::Evaluate(queryPoint, referencePoint));

    const double maxSide = getMaxSide(referenceNode.Bound());
    if (maxSide / distance < theta)
    {
      const double q = 1.0 / (1.0 + distance);
      sumQ += referenceNode.NumDescendants() * q;
      negF.col(oldFromNew[queryIndex]) += referenceNode.NumDescendants() * q *
                                          q * (queryPoint - referencePoint);
      return DBL_MAX;
    }
    else
    {
      return maxSide / distance;
    }
  }

  double Rescore(const size_t queryIndex,
                 TreeType &referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

  double Score(TreeType &queryNode, TreeType &referenceNode)
  {
    const VecType &queryPoint = queryNode.Stat().CenterOfMass();
    const VecType &referencePoint = referenceNode.Stat().CenterOfMass();
    const double distance = std::max(
        arma::datum::eps, DistanceType::Evaluate(queryPoint, referencePoint));

    const double maxSide = std::max(getMaxSide(queryNode.Bound()),
                                    getMaxSide(referenceNode.Bound()));
    if (maxSide / distance < theta)
    {
      const double q = 1.0 / (1.0 + distance);
      sumQ += queryNode.NumDescendants() * referenceNode.NumDescendants() * q;
      for (size_t i = 0; i < queryNode.NumDescendants(); i++)
        negF.col(oldFromNew[queryNode.Descendant(i)]) +=
            referenceNode.NumDescendants() * q * q *
            (queryPoint - referencePoint);
      for (size_t i = 0; i < referenceNode.NumDescendants(); i++)
        negF.col(oldFromNew[referenceNode.Descendant(i)]) +=
            queryNode.NumDescendants() * q * q * (referencePoint - queryPoint);
      return DBL_MAX;
    }
    else
    {
      return maxSide / distance;
    }
  }

  double Rescore(TreeType &queryNode,
                 TreeType &referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

  double getMaxSide(const BoundType &bound)
  {
    double maxSide = 0.0;
    for (size_t i = 0; i < bound.Dim(); i++)
      maxSide = std::max(maxSide, bound[i].Hi() - bound[i].Lo());
    return maxSide;
  }

  class TraversalInfoType
  {
  };
  const TraversalInfoType &TraversalInfo() const { return traversalInfo; }
  TraversalInfoType &TraversalInfo() { return traversalInfo; }

 private:
  double &sumQ;
  MatType &negF;
  const double theta;
  const MatType &embedding;
  const std::vector<size_t> &oldFromNew;
  TraversalInfoType traversalInfo;
};

template <typename TreeType = mlpack::Octree<>,
          typename MatType = arma::mat,
          typename VecType = arma::vec>
class BarnesHutTraversalRule
{
 public:
  BarnesHutTraversalRule(double &Z,
                         VecType &negFi,
                         const MatType &querySet,
                         const MatType &referenceSet,
                         const double theta = 0.5)
      : Z(Z), negFi(negFi), querySet(querySet), referenceSet(referenceSet),
        theta(theta)
  {
    // Nothing To Do Here
  }

  double BaseCase(const size_t queryIndex, const size_t referenceIndex)
  {
    VecType queryPoint, referencePoint;
    queryPoint = querySet.col(queryIndex);
    referencePoint = referenceSet.col(referenceIndex);
    double distance =
        mlpack::SquaredEuclideanDistance::Evaluate(queryPoint, referencePoint);

    if (distance >= arma::datum::eps)
    {
      double q = 1.0 / (1.0 + distance);
      Z += q;
      negFi += q * q * (queryPoint - referencePoint);
    }

    return distance;
  }

  double Score(const size_t queryIndex, TreeType &referenceNode)
  {
    const auto &bound = referenceNode.Bound();

    double max_side = 0.0;
    for (size_t i = 0; i < bound.Dim(); i++)
      max_side = std::max(max_side, bound[i].Hi() - bound[i].Lo());

    VecType queryPoint, referencePoint;
    queryPoint = querySet.col(queryIndex);
    referencePoint = referenceNode.Stat().CenterOfMass();
    double distance = std::max(
        arma::datum::eps,
        mlpack::SquaredEuclideanDistance::Evaluate(queryPoint, referencePoint));

    if (max_side * max_side / distance < theta * theta)
    {
      double q = 1.0 / (1.0 + distance);
      Z += referenceNode.NumDescendants() * q;
      negFi += referenceNode.NumDescendants() * q * q *
               (queryPoint - referencePoint);
      return DBL_MAX;
    }
    else
    {
      return max_side * max_side / distance;
    }
  }

  double Rescore(const size_t queryIndex,
                 TreeType &referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

 private:
  double &Z;
  VecType &negFi;
  const double theta;
  const MatType &querySet;
  const MatType &referenceSet;
};


template <typename MatType = arma::mat, typename VecType = arma::vec>
arma::sp_mat binarySearchPerplexity(const arma::Mat<size_t> &N,
                                    const MatType &D,
                                    double perplexity)
{
  const size_t maxSteps = 100;
  const double tolerance = 1e-5;

  const size_t n = D.n_cols;
  const size_t k = D.n_rows;
  const double hDesired = std::log(perplexity);

  arma::sp_mat P(n, n);
  VecType beta(n, arma::fill::ones);

  VecType Di;
  for (size_t i = 0; i < n; i++)
  {
    double betamin = -arma::datum::inf;
    double betamax = +arma::datum::inf;
    double sumP, sumDP, hDiff, hApprox;

    if (i % 1000 == 0)
      std::cout << "Computing P-values for points " << i + 1 << " To "
                << std::min(n, i + 1000) << std::endl;

    Di = D.col(i);
    arma::sp_vec Pi(n);

    int step = 0;
    while (step < maxSteps)
    {
      sumP = sumDP = 0.0;
      for (size_t j = 0; j < k; j++)
      {
        Pi(N(j, i)) = std::exp(-Di(j) * beta(i));
        sumP += Pi(N(j, i));
        sumDP += Di(j) * Pi(N(j, i));
      }
      sumP = std::max(sumP, arma::datum::eps);
      hApprox = std::log(sumP) + beta(i) * sumDP / sumP;
      Pi /= sumP;

      hDiff = hApprox - hDesired;
      if (std::abs(hDiff) <= tolerance)
        break;

      if (hDiff > 0)
      {
        betamin = beta(i);
        if (std::isinf(betamax))
          beta(i) *= 2.0;
        else
          beta(i) = (beta(i) + betamax) / 2.0;
      }
      else
      {
        betamax = beta(i);
        if (std::isinf(betamin))
          beta(i) /= 2.0;
        else
          beta(i) = (beta(i) + betamin) / 2.0;
      }
      step++;
    }
    if (step == maxSteps)
      std::cout << "Max Steps Reached While Searching For Desired Perplexity"
                << std::endl;

    for (size_t j = 0; j < k; ++j)
      P(i, N(j, i)) = Pi(N(j, i));
  }
  std::cout << "Mean value of sigma: " << arma::mean(arma::sqrt(1.0 / beta))
            << std::endl;

  return P;
}


template <typename MatType = arma::mat, typename VecType = arma::vec>
class TSNE
{
 public:
  TSNE(const MatType &X,
       const size_t dimensions = 2,
       const double perplexity = 20.0)
      : dimensions(dimensions), perplexity(perplexity)
  {
    mlpack::PCA pca;
    pca.Apply(X, Y, dimensions);

    mlpack::NeighborSearch<mlpack::NearestNeighborSort,
                           mlpack::SquaredEuclideanDistance>
        knn(X);

    const size_t ksearch = static_cast<size_t>(3 * perplexity);
    knn.Search(ksearch, N, D);

    P = binarySearchPerplexity(N, D, perplexity);
    P = P + P.t();
    P /= std::max(arma::accu(P), arma::datum::eps);
  }

  double EvaluateWithGradient(const MatType &y,
                              const size_t i,
                              MatType &g,
                              const size_t n)
  {
    std::vector<size_t> oldFromNew, newFromOld;
    mlpack::Octree<mlpack::SquaredEuclideanDistance, CenterOfMassStatistic<>>
        tree(y, oldFromNew, newFromOld, 1);

    double Z = 0.0;
    TSNERules<BarnesHutTSNE,
              SquaredEuclideanDistance,
              mlpack::Octree<mlpack::SquaredEuclideanDistance,
                             CenterOfMassStatistic<>>>
        rule(Z, g, y, oldFromNew);
    mlpack::Octree<mlpack::SquaredEuclideanDistance,
                   CenterOfMassStatistic<>>::SingleTreeTraverser trav(rule);

    for (size_t i = 0; i < g.n_cols; i++)
    {
      trav.Traverse(i, tree);
    }
    g = -g / Z;

    double error = 0.0;
    for (size_t i = 0; i < N.n_cols; i++)
    {
      for (size_t j = 0; j < N.n_rows; j++)
      {
        double q = 1.0 / (1.0 + mlpack::SquaredEuclideanDistance::Evaluate(
                                    y.col(i), y.col(N(j, i))));

        g.col(i) += q * P(i, N(j, i)) * (y.col(i) - y.col(N(j, i)));
        error += P(i, N(j, i)) *
                 std::log(P(i, N(j, i)) / std::max(q / Z, arma::datum::eps));
      }
    }

    g *= 4;
    return n * error;
  }

  void Shuffle() {}
  size_t NumFunctions() { return Y.n_cols; }

  void StartExaggerating() { P *= 4; }
  void StopExaggerating() { P /= 4; }

  MatType Embedding() const { return Y; }
  MatType &Embedding() { return Y; }

 private:
  arma::sp_mat P;
  MatType D, Y, Q;
  arma::Mat<size_t> N;
  const size_t dimensions;
  const double perplexity;
};


class TSNEOptimizationManager
{
 public:
  TSNEOptimizationManager() {};

  template <typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType &optimizer,
                         FunctionType &function,
                         MatType & /* coordinates */)
  {
    function.StartExaggerating();
  }

  // template <typename OptimizerType, typename FunctionType, typename MatType>
  // bool BeginEpoch(OptimizerType & optimizer,
  //                 FunctionType & function,
  //                 const MatType & coordinates,
  //                 const size_t epochIn,
  //                 const double objective)
  // {

  // }

  template <typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType &optimizer,
                FunctionType &function,
                const MatType &coordinates,
                const size_t epoch,
                const double /* objective */)
  {
    if (epoch == 50)
    {
      function.StopExaggerating();
    }
    if (epoch == 250)
    {
      optimizer.UpdatePolicy().Momentum() = 0.8;
    }

    last_grad = coordinates;
    return false;
  }

  template <typename OptimizerType, typename FunctionType, typename MatType>
  bool EvaluateWithGradient(OptimizerType &optimizer,
                            FunctionType &function,
                            const MatType &coordinates,
                            const double /* objective */)
  {
    arma::umat inc = last_grad * coordinates < 0.0;
    arma::umat dec = last_grad * coordinates >= 0.0;
    gains.elem(inc) += 0.2;
    gains.elem(dec) *= 0.8;
    gains.elem(arma::find(gains < 0.01)).fill(0.01);

    // Scale gradient 
    coordinates %= gains;
    return false;
  }

 private:
  arma::mat gains;
  arma::mat last_grad;
};


int main(int argc, char **argv)
{
  arma::mat X;
  if (!mlpack::data::Load("mnist2500_X.txt", X, false))
  {
    std::cerr << "Could not load mnist2500_X.txt!" << std::endl;
    return 1;
  }

  TSNE tsne(X);
  ens::MomentumSGD optimizer(200.0, X.n_cols, 1000 * X.n_cols);

  std::cout << "OPTIMIZING" << std::endl;
  optimizer.Optimize(tsne, tsne.Embedding(), TSNEOptimizationManager(), ens::ProgressBar());
  std::cout << "OPTIMIZED!" << std::endl;

  std::string output_filename = "tsne_result.csv";
  if (!mlpack::data::Save(output_filename, tsne.Embedding(), false))
  {
    std::cerr << "Could not save result to " << output_filename << std::endl;
    return 1;
  }

  std::cout << "t-SNE completed. Results saved to " << output_filename
            << std::endl;

  return 0;
}