import numpy as np
from scipy.special import expit
from numpy.linalg import norm


class Logistic_Regression(object):
    def __init__(self, random_state=None, penalty='l2', solver='sgd'):
        self.random_state = random_state
        self.penalty = penalty
        self.solver = solver
        self.coef_ = 0

    def fit(self, X, y, epsilon=1e-5, alpha=0.5, max_iter=1000, step_size=0.01, verbose=False):
        y_modified = y.copy()
        y_modified = self.check_y(y_modified)
        if self.solver == 'sgd':
            self.coef_, self.intercept_ = _fit_sgd(X, y_modified, epsilon, alpha, max_iter, step_size, verbose)

    def predict_prob(self, X):
        y_pred = sigmoid(X.dot(self.coef_) + self.intercept_)
        return y_pred

    def predict(self, X):
        y_pred = sigmoid(X.dot(self.coef_) + self.intercept_)
        y_pred[y_pred <= 0.5] = 0
        y_pred[y_pred > 0.5] = 1
        return y_pred

    def check_y(self, y):
        term = np.unique(y)
        if term.size != 2:
            raise ValueError('class size should be 2')
        if not np.all(term == np.array([-1, 1])):
            y[y == term[0]] = -1
            y[y == term[1]] = 1
        return y

def sigmoid(x):
    sig = np.empty_like(x)
    sig = 1 / (1 + np.exp(-x))
    return sig


def log_logistic(x):
    out = np.empty_like(x)
    out = np.log(sigmoid(x))
    return out


def get_accuracy(y_pre, y_true):
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    num = y_pre.size
    count = 0
    y_pre = list(y_pre)
    y_true = list(y_true)
    for pre, true in zip(y_pre, y_true):
        if pre == true:
            count += 1
    accuracy = count / num
    return accuracy


def _intercept_dot(w, X, y):
    """Computes y*np.dot(X,w)

    Parameters:
    ----------
    w,X,y

    Returns:
    ---------
    w: ndarrray,shape (n_features,)

    c: float
       The intercept

    yz: float

    """
    c = 0
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]
    z = np.dot(X, w) + c
    yz = y * z
    return w, c, yz


def _logistic_loss_and_grad(w, X, y, alpha):
    """Computes the logistic loss and gradient

    Parameters:
    -----------
    w: ndarray, shape (n_features,) or (n_features+1,)
    X
    y
    alpha: float
            l2 penalty

    Returns:
    -----------
    out: float
        Logistic loss
    grad: ndarray,shape (n_features,) or (n_features+1,)
        Logistic gradient
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    out = -np.sum(log_logistic(yz)) / n_samples + 0.5 * alpha * np.dot(w, w)
    z = expit(yz)
    z0 = (z - 1) * y
    grad[:n_features] = X.T.dot(z0) + alpha * w

    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()

    return out, grad


def _logistic_loss(step_size, grad, w, X, y, alpha):
    """Computes the logistic loss and gradient

    Parameters:
    -----------
    w: ndarray, shape (n_features,) or (n_features+1,)
    X
    y
    alpha: float
            l2 penalty

    Returns:
    -----------
    out: float
        Logistic loss
    grad: ndarray,shape (n_features,) or (n_features+1,)
        Logistic gradient
    """
    n_samples, n_features = X.shape
    w = w - step_size * grad

    w, c, yz = _intercept_dot(w, X, y)

    out = -np.sum(log_logistic(yz)) / n_samples + 0.5 * alpha * np.dot(w, w)

    return out


def _fit_sgd(X, y, epsilon, alpha, max_iter, step_size, verbose):
    n_samples, n_features = X.shape
    w = np.random.random(n_features + 1)
    k = 0
    loss, grad = _logistic_loss_and_grad(w, X, y, alpha)
    if norm(grad, ord=np.inf) < epsilon:
        coef_ = w[:-1]
        intercept_ = w[-1]
        return coef_, intercept_
    w_new = w - step_size * grad
    loss_new, grad_new = _logistic_loss_and_grad(w_new, X, y, alpha)
    err_loss = norm(loss_new - loss)
    err_grad = norm(grad_new - grad, ord=np.inf)
    loss, grad, w = loss_new, grad_new, w_new
    k += 1

    while err_loss >= epsilon and err_grad >= epsilon and k <= max_iter:
        k += 1
        w_new = w - step_size * grad
        loss_new, grad_new = _logistic_loss_and_grad(w_new, X, y, alpha)
        err_loss = norm(loss_new - loss)
        err_grad = norm(grad_new - grad)
        loss, grad, w = loss_new, grad_new, w_new
        if verbose:
            print('-------------------')
            print('Step %d' % k)
            print('err_loss : {:1.10f}'.format(err_loss))
            print('err_grad : {:1.10f}'.format(err_grad))

    coef_ = w[:-1]
    intercept_ = w[-1]
    return coef_, intercept_
