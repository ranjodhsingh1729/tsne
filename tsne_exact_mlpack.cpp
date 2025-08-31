#include <bits/stdc++.h>

#include <armadillo>
#include <ensmallen.hpp>
#include <ensmallen_bits/callbacks/progress_bar.hpp>
#include <mlpack.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>
#include <mlpack/core/distances/lmetric.hpp>


template <typename MatType = arma::mat, typename VecType = arma::vec>
MatType binarySearchPerplexity(const MatType &D, double perplexity) {
  const size_t maxSteps = 50;
  const double tolerance = 1e-5;

  size_t n = D.n_cols;
  double H = std::log(perplexity);
  VecType beta(n, arma::fill::ones);
  MatType P(n, n, arma::fill::zeros);

  VecType Di, Pi;
  double betamin, betamax;
  double sumP, Happrox, Hdiff;
  for (size_t i = 0; i < n; i++) {
    if (i % 1000 == 0)
      std::cout << "Computing P-values for points " << i + 1 << " To "
                << std::min(n, i + 1000) << std::endl;

    betamin = -arma::datum::inf;
    betamax = +arma::datum::inf;
    Di = D.col(i);
    Di.shed_row(i);

    int step = 0;
    while (step < maxSteps) {
      Pi = arma::exp(-Di * beta(i));
      sumP = std::max(arma::datum::eps, arma::accu(Pi));
      Happrox = std::log(sumP) + beta(i) * arma::accu(Di % Pi) / sumP;
      Pi /= sumP;

      Hdiff = Happrox - H;
      if (std::abs(Hdiff) <= tolerance)
        break;

      if (Hdiff > 0) {
        betamin = beta(i);
        if (std::isinf(betamax))
          beta(i) *= 2.0;
        else
          beta(i) = (beta(i) + betamax) / 2.0;
      } else {
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

    P.row(i).head(i) = Pi.head(i).t();
    P.row(i).tail(n - i - 1) = Pi.tail(n - i - 1).t();
  }
  std::cout << "Mean value of sigma: " << arma::mean(arma::sqrt(1.0 / beta))
            << std::endl;

  return P;
}

template <typename MatType = arma::mat, typename VecType = arma::vec>
class TSNE {
public:
  TSNE(size_t dimensions = 2, double perplexity = 20.0)
      : dimensions(dimensions), perplexity(perplexity) {
    // NOTHING TO DO
  }

  void initialize(const MatType &X) {
    // RND
    // Y = arma::mat(dimensions, X.n_cols, arma::fill::randn);
    // PCA
    mlpack::PCA pca;
    pca.Apply(X, Y, dimensions);
    
    // CALC D
    D = mlpack::PairwiseDistances(X, mlpack::SquaredEuclideanDistance());

    // CALC P
    P = binarySearchPerplexity(D, perplexity);

    // SYM AND NORM P
    P = P + P.t();
    P /= arma::accu(P);
    P.elem(arma::find(P < arma::datum::eps)).fill(arma::datum::eps);
  }

  double Evaluate(const arma::mat &x, const size_t i, const size_t n) {
    MatType Q;
    Q = mlpack::PairwiseDistances(x, mlpack::SquaredEuclideanDistance());
    Q = 1.0 / (1.0 + Q);
    Q.diag().zeros();
    Q /= arma::accu(Q);
    Q.elem(arma::find(Q < arma::datum::eps)).fill(arma::datum::eps);
    return arma::accu(P % arma::log(P / Q));
  }

  void Gradient(const arma::mat &x, const size_t i, arma::mat &g,
                const size_t n) {
    MatType q, Q;
    q = mlpack::PairwiseDistances(x, mlpack::SquaredEuclideanDistance());
    q = 1.0 / (1.0 + q);
    q.diag().zeros();
    Q = q / arma::accu(q);
    Q.elem(arma::find(Q < arma::datum::eps)).fill(arma::datum::eps);
    MatType PQ = P - Q;

    for (size_t i = 0; i < n; i++) {
      VecType mult = PQ.col(i) % q.col(i);
      MatType diff = arma::repmat(Y.col(i), 1, n) - Y;
      MatType term = diff % arma::repmat(mult.t(), Y.n_rows, 1);
      g.col(i) = arma::sum(term, 1);
    }
  }

  double EvaluateWithGradient(const MatType &x, const size_t i, MatType &g,
                              const size_t n) {
    MatType q, Q;
    q = mlpack::PairwiseDistances(x, mlpack::SquaredEuclideanDistance());
    q = 1.0 / (1.0 + q);
    q.diag().zeros();
    Q = q / arma::accu(q);
    Q.elem(arma::find(Q < arma::datum::eps)).fill(arma::datum::eps);
    MatType PQ = P - Q;

    for (size_t i = 0; i < n; i++) {
      VecType mult = PQ.col(i) % q.col(i);
      MatType diff = arma::repmat(Y.col(i), 1, n) - Y;
      MatType term = diff % arma::repmat(mult.t(), Y.n_rows, 1);
      g.col(i) = arma::sum(term, 1);
    }

    return arma::accu(P % arma::log(P / Q)); // ERROR
  }

  void Shuffle() {
    // DO NOTHING
  }

  size_t NumFunctions() { return Y.n_cols; }

  size_t dimensions;
  double perplexity;
  MatType Y, D, P, Q;
};

int main(int argc, char **argv) {
  arma::mat X;
  if (!mlpack::data::Load("mnist2500_X.txt", X, false)) {
    std::cerr << "Could not load mnist2500_X.txt!" << std::endl;
    return 1;
  }

  TSNE tsne;
  tsne.initialize(X);

  ens::MomentumSGD optimizer1(500.0, X.n_cols, 20 * X.n_cols, 1e-5, false,
                              ens::MomentumUpdate(0.5));
  ens::MomentumSGD optimizer2(500.0, X.n_cols, 80 * X.n_cols, 1e-5, false,
                              ens::MomentumUpdate(0.8));
  ens::MomentumSGD optimizer3(500.0, X.n_cols, 220 * X.n_cols, 1e-5, false,
                              ens::MomentumUpdate(0.8));

  std::cout << "OPTIMIZING" << std::endl;
  tsne.P *= 4;
  optimizer1.Optimize(tsne, tsne.Y, ens::ProgressBar());
  optimizer2.Optimize(tsne, tsne.Y, ens::ProgressBar());
  tsne.P /= 4;
  optimizer3.Optimize(tsne, tsne.Y, ens::ProgressBar());
  std::cout << "OPTIMIZED!" << std::endl;

  std::string output_filename = "tsne_result.csv";
  if (!mlpack::data::Save(output_filename, tsne.Y, false)) {
    std::cerr << "Could not save result to " << output_filename << std::endl;
    return 1;
  }

  std::cout << "t-SNE completed. Results saved to " << output_filename
            << std::endl;

  return 0;
}