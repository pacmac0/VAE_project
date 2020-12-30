import torch

min_epsilon = 1e-5
max_epsilon = 1.0 - 1e-5


def log_Normal_standard(x, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    return torch.sum(log_normal, dim)


def log_Normal_diag(x, mean, log_var, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    return torch.sum(log_normal, dim)


def log_Bernoulli(x, mean, dim=None):
    probs = torch.clamp(mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1.0 - x) * torch.log(1.0 - probs)
    return torch.sum(log_bernoulli, dim)


def log_Logistic_256(x, mean, logvar, reduce=True, dim=None):
    bin_size = 1.0 / 256.0

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size / scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = -torch.log(cdf_plus - cdf_minus + 1.0e-7)

    return torch.sum(log_logist_256, dim) if reduce else log_logist_256
