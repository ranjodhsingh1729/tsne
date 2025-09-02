#include <bits/stdc++.h>

#include <armadillo>
#include <ensmallen.hpp>
#include <ensmallen_bits/callbacks/progress_bar.hpp>
#include <mlpack.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/tree/octree/octree.hpp>
#include <mlpack/core/tree/statistic.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>


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


template <typename TreeType = mlpack::Octree<>,
          typename MatType = arma::mat,
          typename VecType = arma::vec>
class DualTreeTraversalRule
{
 public:
  DualTreeTraversalRule(double &Z,
                        MatType &negF,
                        const MatType &querySet,
                        const MatType &referenceSet,
                        const std::vector<size_t> &oldFromNew,
                        const std::vector<size_t> &newFromOld,
                        const double theta = 0.5)
      : Z(Z), negF(negF), querySet(querySet), referenceSet(referenceSet),
        oldFromNew(oldFromNew), newFromOld(newFromOld), theta(theta)
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
      negF.col(oldFromNew[queryIndex]) += q * q * (queryPoint - referencePoint);
      negF.col(oldFromNew[referenceIndex]) += q * q * (referencePoint - queryPoint);
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
      negF.col(oldFromNew[queryIndex]) += referenceNode.NumDescendants() * q * q *
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

  double Score(TreeType &queryNode, TreeType &referenceNode)
  {
    const auto &queryBound = queryNode.Bound();
    const auto &referenceBound = referenceNode.Bound();

    double queryMaxSide = 0.0;
    for (size_t i = 0; i < queryBound.Dim(); i++)
      queryMaxSide =
          std::max(queryMaxSide, queryBound[i].Hi() - queryBound[i].Lo());

    double referenceMaxSide = 0.0;
    for (size_t i = 0; i < referenceBound.Dim(); i++)
      referenceMaxSide = std::max(
          referenceMaxSide, referenceBound[i].Hi() - referenceBound[i].Lo());

    VecType queryPoint, referencePoint;
    queryPoint = queryNode.Stat().CenterOfMass();
    referencePoint = referenceNode.Stat().CenterOfMass();
    double distance = std::max(
        arma::datum::eps,
        mlpack::SquaredEuclideanDistance::Evaluate(queryPoint, referencePoint));

    double maxSide = std::max(queryMaxSide, referenceMaxSide);
    if (maxSide * maxSide / distance < theta * theta)
    {
      double q = 1.0 / (1.0 + distance);
      Z += queryNode.NumDescendants() * referenceNode.NumDescendants() * q;

      // THERE GOES THE ADVANTAGE
      for (size_t i = 0; i < queryNode.NumDescendants(); i++)
      {
        negF.col(oldFromNew[queryNode.Descendant(i)]) +=
            referenceNode.NumDescendants() * q * q *
            (queryPoint - referencePoint);
      }
      for (size_t i = 0; i < referenceNode.NumDescendants(); i++)
      {
        negF.col(oldFromNew[referenceNode.Descendant(i)]) +=
            queryNode.NumDescendants() * q * q * (referencePoint - queryPoint);
      }

      return DBL_MAX;
    }
    else
    {
      return maxSide * maxSide / distance;
    }
  }

  double Rescore(TreeType &queryNode,
                 TreeType &referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

  class TraversalInfoType
  {
  };
  const TraversalInfoType &TraversalInfo() const { return traversalInfo; }
  TraversalInfoType &TraversalInfo() { return traversalInfo; }

 private:
  double &Z;
  MatType &negF;
  const double theta;
  const MatType &querySet;
  const MatType &referenceSet;
  const std::vector<size_t> oldFromNew;
  const std::vector<size_t> newFromOld;
  TraversalInfoType traversalInfo;
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
       const double perplexity = 30.0)
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

  // double Evaluate(const arma::mat &y, const size_t i, const size_t n) {}

  // void Gradient(const arma::mat &y,
  //               const size_t i,
  //               arma::mat &g,
  //               const size_t n)
  // {
  // }

  double EvaluateWithGradient(const MatType &y,
                              const size_t i,
                              MatType &g,
                              const size_t n)
  {
    std::vector<size_t> oldFromNew, newFromOld;
    mlpack::Octree<mlpack::SquaredEuclideanDistance, CenterOfMassStatistic<>>
        tree(y, oldFromNew, newFromOld, 1);

    double Z = 0.0;
    DualTreeTraversalRule<mlpack::Octree<mlpack::SquaredEuclideanDistance,
                                         CenterOfMassStatistic<>>>
        rule(Z, g, y, y, oldFromNew, newFromOld);
    mlpack::Octree<mlpack::SquaredEuclideanDistance,
                   CenterOfMassStatistic<>>::DualTreeTraverser trav(rule);
    trav.Traverse(tree, tree);
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


int main(int argc, char **argv)
{
  arma::mat X;
  if (!mlpack::data::Load("mnist2500_X.txt", X, false))
  {
    std::cerr << "Could not load mnist2500_X.txt!" << std::endl;
    return 1;
  }

  TSNE tsne(X);
  ens::MomentumSGD optimizer1(
      200.0, X.n_cols, 20 * X.n_cols, 1e-5, false, ens::MomentumUpdate(0.5));
  ens::MomentumSGD optimizer2(
      200.0, X.n_cols, 80 * X.n_cols, 1e-5, false, ens::MomentumUpdate(0.8));
  ens::MomentumSGD optimizer3(
      200.0, X.n_cols, 1920 * X.n_cols, 1e-5, false, ens::MomentumUpdate(0.8));

  std::cout << "OPTIMIZING" << std::endl;
  tsne.StartExaggerating();
  optimizer1.Optimize(tsne, tsne.Embedding(), ens::ProgressBar());
  optimizer2.Optimize(tsne, tsne.Embedding(), ens::ProgressBar());
  tsne.StopExaggerating();
  optimizer3.Optimize(tsne, tsne.Embedding(), ens::ProgressBar());
  std::cout << "OPTIMIZED!" << std::endl;

  std::string output_filename = "tsne_result_again.csv";
  if (!mlpack::data::Save(output_filename, tsne.Embedding(), false))
  {
    std::cerr << "Could not save result to " << output_filename << std::endl;
    return 1;
  }

  std::cout << "t-SNE completed. Results saved to " << output_filename
            << std::endl;

  return 0;
}