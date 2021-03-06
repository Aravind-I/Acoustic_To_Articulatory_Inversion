from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture import GaussianMixture
import numpy as np
class GMMBase_M(object):
    def __init__(self, gmm, swap=False, diff=False):
        assert gmm.covariance_type == "full"
        # D: static + delta dim
        Dx = 39#gmm.means_.shape[1] // 2
        Dy=36
        self.num_mixtures = gmm.means_.shape[0]
        self.weights = gmm.weights_

        # Split source and target parameters from joint GMM
        self.src_means = gmm.means_[:, :Dx]
        self.tgt_means = gmm.means_[:, Dx:]
        self.covarXX = gmm.covariances_[:, :Dx, :Dx]
        self.covarXY = gmm.covariances_[:, :Dx, Dx:]
        self.covarYX = gmm.covariances_[:, Dx:, :Dx]
        self.covarYY = gmm.covariances_[:, Dx:, Dx:]
        
        self.px = GaussianMixture(
            n_components=self.num_mixtures, covariance_type="full")
        self.px.means_ = self.src_means
        self.px.covariances_ = self.covarXX
        self.px.weights_ = self.weights
        self.px.precisions_cholesky_ = _compute_precision_cholesky(
            self.px.covariances_, "full")

    def transform(self, src):
        if src.ndim == 2:
            tgt = np.zeros_like(src)
            for idx, x in enumerate(src):
                y = self._transform_frame(x)
                tgt[idx][:len(y)] = y
            return tgt
        else:
            return self._transform_frame(src)

    def _transform_frame(self, src):
        """Mapping source spectral feature x to target spectral feature y
        so that minimize the mean least squared error.
        More specifically, it returns the value E(p(y|x)].

        Args:
            src (array): shape (`order of spectral feature`) source speaker's
                spectral feature that will be transformed

        Returns:
            array: converted spectral feature
        """
        D = len(src)

        # Eq.(11)
        E = np.zeros((self.num_mixtures, D))
        for m in range(self.num_mixtures):
            xx = np.linalg.solve(self.covarXX[m], src - self.src_means[m])
            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx)

        # Eq.(9) p(m|x)
        posterior = self.px.predict_proba(np.atleast_2d(src))

        # Eq.(13) conditinal mean E[p(y|x)]
        return posterior.dot(E).flatten()


class GMM_M(GMMBase_M):
    """Maximum likelihood Parameter Generation (MLPG) for GMM-basd voice
    conversion [1].
    
    .. [1] [Toda 2007] Voice Conversion Based on Maximum Likelihood Estimation
      of Spectral Parameter Trajectory.
    """

    def __init__(self, gmm, windows=None, swap=False, diff=False):
        super(GMM_M, self).__init__(gmm, swap, diff)
        if windows is None:
            windows = [
                (0, 0, np.array([1.0])),
                (1, 1, np.array([-0.5, 0.0, 0.5])),
            ]
        self.windows = windows
        self.Y_static_dim = 12 #self.static_dim = gmm.means_.shape[-1] // 2 // len(windows)

    def transform(self, src):
        """Mapping source feature x to target feature y so that maximize the
        likelihood of y given x.

        Args:
            src (array): shape (`the number of frames`, `the order of spectral
                feature`) a sequence of source speaker's spectral feature that
                will be transformed.

        Returns:
            array: a sequence of transformed features
        """
        T, feature_dim = src.shape[0], self.Y_static_dim*3

        if feature_dim == self.Y_static_dim:
            return super(GMM_M, self).transform(src)

        # A suboptimum mixture sequence  (eq.37)
        optimum_mix = self.px.predict(src)

        # Compute E eq.(40)
        E = np.empty((T, feature_dim))
        for t in range(T):
            m = optimum_mix[t]  # estimated mixture index at time t
            xx = np.linalg.solve(self.covarXX[m], src[t] - self.src_means[m])
            #print(xx.shape,self.tgt_means[m].shape,self.covarYX[m].shape)
            # Eq. (22)
            E[t] = self.tgt_means[m] +np.dot(self.covarYX[m], xx)

        # Compute D eq.(23)
        # Approximated variances with diagonals so that we can do MLPG
        # efficiently in dimention-wise manner
        #print(E.shape)
        D = np.empty((T, feature_dim))
        #print(D.shape)
        for t in range(T):
            m = optimum_mix[t]
            # Eq. (23), with approximating covariances as diagonals
            #D[t] = np.diag(self.covarYY[m]) - np.diag(self.covarYX[m]) / \
             #   np.diag(self.covarXX[m]) * np.diag(self.covarXY[m])

            # Exact Inference
            dd = self.covarYY[m] - np.linalg.multi_dot([self.covarYX[m], np.linalg.pinv(self.covarXX[m]), self.covarXY[m]])
            #print(dd.shape)
            D[t] = np.diag(dd)

        # Once we have mean and variance over frames, then we can do MLPG
        return E, D, self.windows#mlpg(E, D, self.windows)
