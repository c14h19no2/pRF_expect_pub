import numpy as np
import scipy.stats as stats
from scipy.stats import t


def design_variance(X, contrast_vector):
    """Returns the design variance of a predictor (or contrast) in X.

    Parameters
    ----------
    X : numpy array
        Array of shape (N, P)
    contrast_vector : list/array
        A contrast-vector

    Returns
    -------
    des_var : float
        Design variance of the specified predictor/contrast from X.
    """

    idx = np.array(contrast_vector) != 0
    contrast_vector = np.array(contrast_vector)

    c = np.zeros(X.shape[1])
    c[idx] = contrast_vector[idx]
    des_var = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)
    return des_var


def calc_t_stat(Xn, y, yhat_meter, beta_meter, contrast_vector):
    """Calculate the t-statistic for a predictor/contrast.
    
    Parameters
    ----------
    Xn : numpy array
        Design matrix of shape (N, P)
    y : numpy array
        Observed data of shape (N,)
    yhat_meter : numpy array
        Predicted data of shape (N,)
    beta_meter : numpy array
        Estimated beta coefficients of shape (P,)
    contrast_vector : list/array
        A contrast-vector
    """
    N = yhat_meter.size
    P = Xn.shape[1]
    df = N - P
    sigma_hat = np.sum((y - yhat_meter) ** 2) / df
    design_variance_weight = design_variance(Xn, contrast_vector)
    t_meter = (beta_meter * contrast_vector).sum() / np.sqrt(sigma_hat * design_variance_weight)
    p_value = t.sf(np.abs(t_meter), df) * 2  # two-tailed
    return t_meter, p_value


def design_variance_surprise_ratio(X, betas_spar, betas_omit):
    # Compute the coefficients for the contrast vector
    # 1. Define the Ratio Function:
    # Let g(β) = β1/(β1+β2)−0.5

    # 2. Compute the Gradient:
    # The Delta method requires the gradient (first derivatives) of g(β)g(β) with respect to β1β1​ and β2β2​:
    # ∂g/∂β1 = β2/((β1+β2)^2)
    # ∂g/∂β2 = −β1/((β1+β2)^2)

    # 3. Construct the Gradient Vector:

    # The gradient vector ∇g(β)∇g(β) is then:
    # ∇g(β) = [∂g/∂β1, ∂g/∂β2]

    # 4. Compute the Design Variance:
    # The design variance of the ratio can be computed as:
    # Var(g(β))=∇g(β)T⋅Σ⋅∇g(β)

    # where Σ=Var(β) is the covariance matrix of the betas. In the case of GLM, this covariance matrix is often given by σ2(XTX)−1σ2(XTX)−1.
    denominator = (betas_omit + betas_spar) ** 2
    c_omit = -betas_omit / denominator
    c_spar = betas_spar / denominator

    # Create the contrast vector
    c = np.array([c_omit, c_spar])

    # Compute the design variance using the contrast vector
    des_var = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)
    return des_var


def calc_betaratio_t_stat(
    Xn,
    y,
    yhat_meter,
    surprise_ratio,
    betas_spar,
    betas_omit,
):
    N = y.size
    P = Xn.shape[1]
    df = N - P
    design_variance_ratio = design_variance_surprise_ratio(Xn, betas_spar, betas_omit)
    sigma_hat = np.sum((y - yhat_meter) ** 2) / df
    # Compute the t-statistic
    t_stat = surprise_ratio / np.sqrt(sigma_hat * design_variance_ratio)
    p_value = t.sf(np.abs(t_stat), df) * 2  # two-tailed
    return t_stat, p_value
