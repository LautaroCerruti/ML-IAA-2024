import warnings

from abc import ABCMeta, abstractmethod


import numpy as np
from scipy.special import logsumexp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.utils import deprecated
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.utils.validation import _check_sample_weight


__all__ = [
    "CategoricalNNB",
]


class _BaseNNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for not naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X

        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape (n_samples, n_classes).

        predict, predict_proba, and predict_log_proba pass the input through
        _check_X and handle it over to _joint_log_likelihood.
        """

    @abstractmethod
    def _check_X(self, X):
        """To be overridden in subclasses with the actual checks.

        Only used in predict* methods.
        """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

_ALPHA_MIN = 1e-10


class _BaseDiscreteNNB(_BaseNNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per _BaseNB
    _update_feature_log_prob(alpha)
    _count(X, Y)
    """

    @abstractmethod
    def _count(self, X, Y):
        """Update counts that are used to calculate probabilities.

        The counts make up a sufficient statistic extracted from the data.
        Accordingly, this method is called each time `fit` or `partial_fit`
        update the model. `class_count_` and `feature_count_` must be updated
        here along with any model specific counts.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Y : ndarray of shape (n_samples, n_classes)
            Binarized class labels.
        """

    @abstractmethod
    def _update_feature_log_prob(self, alpha):
        """Update feature log probabilities based on counts.

        This method is called each time `fit` or `partial_fit` update the
        model.

        Parameters
        ----------
        alpha : float
            smoothing parameter. See :meth:`_check_alpha`.
        """

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        return self._validate_data(X, accept_sparse="csr", reset=False)

    def _check_X_y(self, X, y, reset=True):
        """Validate X and y in fit methods."""
        return self._validate_data(X, y, accept_sparse="csr", reset=reset)

    def _update_class_log_prior(self, class_prior=None):
        """Update class log priors.

        The class log priors are based on `class_prior`, class count or the
        number of classes. This method is called each time `fit` or
        `partial_fit` update the model.
        """
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                # silence the warning when count is 0 because class was not yet
                # observed
                warnings.simplefilter("ignore", RuntimeWarning)
                log_class_count = np.log(self.class_count_)

            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        if np.min(self.alpha) < 0:
            raise ValueError(
                "Smoothing parameter alpha = %.1e. alpha should be > 0."
                % np.min(self.alpha)
            )
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape[0] == self.n_features_in_:
                raise ValueError(
                    "alpha should be a scalar or a numpy array with shape [n_features]"
                )
        if np.min(self.alpha) < _ALPHA_MIN:
            warnings.warn(
                "alpha too small will result in numeric errors, setting alpha = %.1e"
                % _ALPHA_MIN
            )
            return np.maximum(self.alpha, _ALPHA_MIN)
        return self.alpha

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._check_X_y(X, y)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def _more_tags(self):
        return {"poor_score": True}

    # TODO: Remove in 1.2
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `n_features_` was deprecated in version 1.0 and will be "
        "removed in 1.2. Use `n_features_in_` instead."
    )
    @property
    def n_features_(self):
        return self.n_features_in_

class CategoricalNNB(_BaseDiscreteNNB):
    """Naive Bayes classifier for categorical features.

    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The categories of
    each feature are drawn from a categorical distribution.

    Read more in the :ref:`User Guide <categorical_naive_bayes>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    min_categories : int or array-like of shape (n_features,), default=None
        Minimum number of categories per feature.

        - integer: Sets the minimum number of categories per feature to
          `n_categories` for each features.
        - array-like: shape (n_features,) where `n_categories[i]` holds the
          minimum number of categories for the ith column of the input.
        - None (default): Determines the number of categories automatically
          from the training data.

        .. versionadded:: 0.24

    Attributes
    ----------
    category_count_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the number of samples
        encountered for each class and category of the specific feature.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_log_prob_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the empirical log probability
        of categories given the respective feature and class, ``P(x_i|y)``.

    n_features_ : int
        Number of features of each sample.

        .. deprecated:: 1.0
            Attribute `n_features_` was deprecated in version 1.0 and will be
            removed in 1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_categories_ : ndarray of shape (n_features,), dtype=np.int64
        Number of categories for each feature. This value is
        inferred from the data or set by the minimum number of categories.

        .. versionadded:: 0.24

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    ComplementNB : Complement Naive Bayes classifier.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import CategoricalNB
    >>> clf = CategoricalNB()
    >>> clf.fit(X, y)
    CategoricalNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    def __init__(
        self, *, alpha=1.0, fit_prior=True, class_prior=None, min_categories=None
    ):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().fit(X, y, sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = self._validate_data(
            X, dtype="int", accept_sparse=False, force_all_finite=True, reset=False
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X

    def _check_X_y(self, X, y, reset=True):
        X, y = self._validate_data(
            X, y, dtype="int", accept_sparse=False, force_all_finite=True, reset=reset
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X, y

    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]

    @staticmethod
    def _validate_n_categories(X, min_categories):
        # rely on max for n_categories categories are encoded between 0...n-1
        n_categories_X = X.max(axis=0) + 1
        min_categories_ = np.array(min_categories)
        if min_categories is not None:
            if not np.issubdtype(min_categories_.dtype, np.signedinteger):
                raise ValueError(
                    "'min_categories' should have integral type. Got "
                    f"{min_categories_.dtype} instead."
                )
            n_categories_ = np.maximum(n_categories_X, min_categories_, dtype=np.int64)
            if n_categories_.shape != n_categories_X.shape:
                raise ValueError(
                    f"'min_categories' should have shape ({X.shape[1]},"
                    ") when an array-like is provided. Got"
                    f" {min_categories_.shape} instead."
                )
            return n_categories_
        else:
            return n_categories_X

    def _count(self, X, Y):
        def _update_cat_count_dims(cat_count, highest_feature):
            diff = highest_feature + 1 - cat_count.shape[1]
            if diff > 0:
                # we append a column full of zeros for each new category
                return np.pad(cat_count, [(0, 0), (0, diff)], "constant")
            return cat_count

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            for j in range(n_classes):
                mask = Y[:, j].astype(bool)
                if Y.dtype.type == np.int64:
                    weights = None
                else:
                    weights = Y[mask, j]
                counts = np.bincount(X_feature[mask], weights=weights)
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]

        self.class_count_ += Y.sum(axis=0)
        self.n_categories_ = self._validate_n_categories(X, self.min_categories)
        for i in range(self.n_features_in_):
            X_feature = X[:, i]
            self.category_count_[i] = _update_cat_count_dims(
                self.category_count_[i], self.n_categories_[i] - 1
            )
            _update_cat_count(
                X_feature, Y, self.category_count_[i], self.class_count_.shape[0]
            )

    def _update_feature_log_prob(self, alpha):
        feature_log_prob = []
        for i in range(self.n_features_in_):
            smoothed_cat_count = self.category_count_[i] + alpha
            smoothed_class_count = smoothed_cat_count.sum(axis=1)
            feature_log_prob.append(
                np.log(smoothed_cat_count) - np.log(smoothed_class_count.reshape(-1, 1))
            )
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        self._check_n_features(X, reset=False)
        jll = np.zeros((X.shape[0], self.class_count_.shape[0]))
        for i in range(self.n_features_in_):
            indices = X[:, i]
            jll += self.feature_log_prob_[i][:, indices].T
        total_ll = jll + self.class_log_prior_
        return total_ll 
