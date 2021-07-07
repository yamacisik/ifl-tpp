import torch
import torch.nn as nn

import torch.distributions as D

from data import Batch
from torch.distributions import Categorical

from torch.distributions import Normal as TorchNormal
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions import TransformedDistribution as TorchTransformedDistribution
from util import diff, clamp_preserve_gradients


class MixtureSameFamily(TorchMixtureSameFamily):
    def log_cdf(self, x):
        x = self._pad(x)
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x):
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)


class Normal(TorchNormal):
    def log_cdf(self, x):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x):
        # No numerically stable implementation of log survival is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)


class TransformedDistribution(TorchTransformedDistribution):
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args=validate_args)
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        self.sign = int(sign)

    def log_cdf(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_cdf(x)
        else:
            return self.base_dist.log_survival_function(x)

    def log_survival_function(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_survival_function(x)
        else:
            return self.base_dist.log_cdf(x)


class RecurrentTPP(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
            self,
            num_marks: int,
            mean_log_inter_time: float = 0.0,
            std_log_inter_time: float = 1.0,
            context_size: int = 32,
            mark_embedding_size: int = 32,
            rnn_type: str = "GRU",
    ):
        super().__init__()
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.mark_embedding_size = mark_embedding_size
        if self.num_marks > 1:
            self.num_features = 1 + self.mark_embedding_size
            self.mark_embedding = nn.Embedding(self.num_marks, self.mark_embedding_size)
            self.mark_linear = nn.Linear(self.context_size, self.num_marks)
        else:
            self.num_features = 1
        self.rnn_type = rnn_type
        self.context_init = nn.Parameter(torch.zeros(context_size))  # initial state of the RNN
        self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)

    def get_features(self, batch: Batch) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        features = torch.log(batch.inter_times + 1e-8).unsqueeze(-1)  # (batch_size, seq_len, 1)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        if self.num_marks > 1:
            mark_emb = self.mark_embedding(batch.marks)  # (batch_size, seq_len, mark_embedding_size)
            features = torch.cat([features, mark_emb], dim=-1)
        return features  # (batch_size, seq_len, num_features)

    def get_context(self, features: torch.Tensor, remove_last: bool = True) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
            remove_last: Whether to remove the context embedding for the last event.

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        context = self.rnn(features)[0]
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)
        return context

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raise NotImplementedError()

    def log_prob(self, batch: Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        features = self.get_features(batch)
        context = self.get_context(features)
        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        # last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        # log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        # log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= batch.mask  # (batch_size, seq_len)

        return log_p.sum(-1)

        # return log_p.sum(-1) + log_surv_last  # (batch_size,)

    def sample(self, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None) -> Batch:
        """Generate a batch of sequence from the model.

        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        if context_init is None:
            # Use the default context vector
            context_init = self.context_init
        else:
            # Use the provided context vector
            context_init = context_init.view(self.context_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        inter_times = torch.empty(batch_size, 0)
        if self.num_marks > 1:
            marks = torch.empty(batch_size, 0, dtype=torch.long)

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(next_context)
            next_inter_times = inter_time_dist.sample()  # (batch_size, 1)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (batch_size, seq_len)

            # Generate marks, if necessary
            if self.num_marks > 1:
                mark_logits = torch.log_softmax(self.mark_linear(next_context), dim=-1)  # (batch_size, 1, num_marks)
                mark_dist = Categorical(logits=mark_logits)
                next_marks = mark_dist.sample()  # (batch_size, 1)
                marks = torch.cat([marks, next_marks], dim=1)
            else:
                marks = None

            with torch.no_grad():
                generated = inter_times.sum(-1).min() >= t_end
            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), marks=marks)
            features = self.get_features(batch)  # (batch_size, seq_len, num_features)
            context = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)

        arrival_times = inter_times.cumsum(-1)  # (batch_size, seq_len)
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        if self.num_marks > 1:
            marks = marks * mask  # (batch_size, seq_len)
        return Batch(inter_times=inter_times, mask=mask, marks=marks)


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """

    def __init__(
            self,
            locs: torch.Tensor,
            log_scales: torch.Tensor,
            log_weights: torch.Tensor,
            mean_log_inter_time: float = 0.0,
            std_log_inter_time: float = 1.0
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a ** 2 * variance).logsumexp(-1).exp()


class LogNormMix(RecurrentTPP):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    The distribution of the inter-event times given the history is modeled with a LogNormal mixture distribution.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        num_mix_components: Number of mixture components in the inter-event time distribution.
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
            self,
            num_marks: int,
            mean_log_inter_time: float = 0.0,
            std_log_inter_time: float = 1.0,
            context_size: int = 32,
            mark_embedding_size: int = 32,
            num_mix_components: int = 16,
            rnn_type: str = "GRU",
    ):
        super().__init__(
            num_marks=num_marks,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            context_size=context_size,
            mark_embedding_size=mark_embedding_size,
            rnn_type=rnn_type,
        )
        self.num_mix_components = num_mix_components
        self.linear = nn.Linear(self.context_size, 3 * self.num_mix_components)

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.linear(context)  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 1.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )
