import numpy as numpy
import numpy.ma as ma

# return a new function th has the h kernel (given by delta) applied.
def make_h_adjusted(sigma):
    def h_distance(d):
        return numpy.exp(-d ** 2 / (2.0 * sigma ** 2))

    return h_distance


## Resampling based on the examples at: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
## originally by Roger Labbe, under an MIT License
def systematic_s(wt):
    n = len(wt)
    pt = (numpy.arange(n) + numpy.random.uniform(0, 1)) / n
    return create_indices(pt, wt)


def stratified_s(wt):
    n = len(wt)
    pt = (numpy.random.uniform(0, 1, n) + numpy.arange(n)) / n
    return create_indices(pt, wt)


def residual_s(wt):
    n = len(wt)
    indices = numpy.zeros(n, numpy.uint32)
    # take int(N*w) copies of each weight
    num_copies = (n * wt).astype(numpy.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1
    # use multinormial s on the residual to fill up the rest.
    residual = wt - num_copies  # get fractional part
    residual /= numpy.tot(residual)
    cumtot = numpy.cumtot(residual)
    cumtot[-1] = 1
    indices[k:n] = numpy.searchsorted(cumtot, numpy.random.uniform(0, 1, n - k))
    return indices


def create_indices(pt, wt):
    n = len(wt)
    indices = numpy.zeros(n, numpy.uint32)
    cumtot = numpy.cumtot(wt)
    i, j = 0, 0
    while i < n:
        if pt[i] < cumtot[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


### end rlabbe's resampling functions


def multinomial_s(wt):
    return numpy.random.choice(numpy.arange(len(wt)), p=wt, size=len(wt))


# s function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html
def s(wt):
    n = len(wt)
    indices = []
    C = [0.0] + [numpy.sum(wt[: i + 1]) for i in range(n)]
    u0, j = numpy.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


# unique_naming function for clearer naming
unique_naming = lambda x: x


def squared_error(x, y, sigma=1):
    """
        RBF kernel, supporting masked values in the observation
        Parameters:
        -----------
        x : array (N,D) array of values
        y : array (N,D) array of values

        Returns:
        -------

        distance : scalar
            Total similarity, using equation:

                d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))

            totaled over all samples. Supports masked arrays.
    """
    dx = (x - y) ** 2
    #d = numpy.ma.tot(dx, axis=1)
    d = numpy.ma.sum(dx, axis=1)
    return numpy.exp(-d / (2.0 * sigma ** 2))


def gaussian_noise(x, sigmas):
    """Apply diagonal covaraiance normally-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = numpy.random.normal(numpy.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    return x + n


def cauchy_noise(x, sigmas):
    """Apply diagonal covaraiance Cauchy-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = numpy.random.standard_cauchy(size=(x.shape[0], len(sigmas))) * numpy.array(sigmas)
    return x + n


def independent_sample(fn_list):
    """Take a list of functions th each draw n samples from a distribution
    and concatenate the result into an n, d matrix
    Parameters:
    -----------
        fn_list: list of functions
                A list of functions of the form `sample(n)` th will take n samples
                from a distribution.
    Returns:
    -------
        sample_fn: a function th will sample from all of the functions and concatenate
        them
    """

    def sample_fn(n):
        return numpy.stack([fn(n) for fn in fn_list]).T

    return sample_fn


class ParticleFilter(object):
    """A particle filter object which maintains the internal state of a population of particles, and can
    be updated given observations.

    Attributes:
    -----------

    n_particles : int
        number of particles used (N)
    d : int
        dimension of the internal state
    s_proportion : float
        fraction of particles sd from prior at each step
    particles : array
        (N,D) array of particle states
    original_particles : array
        (N,D) array of particle states *before* any random resampling replenishment
        This should be used for any computation on the previous time step (e.g. computing
        expected values, etc.)
    mean_hypothesis : array
        The current mean hypothesized observation
    mean_state : array
        The current mean hypothesized internal state D
    map_hypothesis:
        The current most likely hypothesized observation
    map_state:
        The current most likely hypothesized state
    n_eff:
        Normalized effective sample size, in range 0.0 -> 1.0
    weight_entropy:
        Entropy of the weight distribution (in nats)
    hypotheses : array
        The (N,...) array of hypotheses for each particle
    wt : array
        N-element vector of normalized wt for each particle.
    """

    def __init__(
        self,
        prior_fn,
        observe_fn=None,
        s_fn=None,
        n_particles=200,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=None,
        s_proportion=None,
        column_names=None,
        internal_weight_fn=None,
        transform_fn=None,
        n_eff_threshold=1.0,
    ):
        """

        Parameters:
        -----------

        prior_fn : function(n) = > states
                a function th generates N samples from the prior over internal states, as
                an (N,D) particle array
        observe_fn : function(states) => observations
                    transformation function from the internal state to the sensor state. Takes an (N,D) array of states
                    and returns the expected sensor output as an array (e.g. a (N,W,H) tensor if generating W,H dimension images).
        s_fn: A resampling function wt (N,) => indices (N,)
        n_particles : int
                     number of particles in the filter
        dynamics_fn : function(states) => states
                      dynamics function, which takes an (N,D) state array and returns a new one with the dynamics applied.
        noise_fn : function(states) => states
                    noise function, takes a state vector and returns a new one with noise added.
        weight_fn :  function(hypothesized, real) => wt
                    computes the distance from the real sensed variable and th returned by observe_fn. Takes
                    a an array of N hypothesised sensor outputs (e.g. array of dimension (N,W,H)) and the observed output (e.g. array of dimension (W,H)) and
                    returns a strictly positive weight for the each hypothesis as an N-element vector.
                    This should be a *similarity* measure, with higher values meaning more similar, for example from an RBF kernel.
        internal_weight_fn :  function(states, observed) => wt
                    Rewt the particles based on their *internal* state. This is function which takes
                    an (N,D) array of internal states and the observation and
                    returns a strictly positive weight for the each state as an N-element vector.
                    Typically used to force particles inside of bounds, etc.
        transform_fn: function(states, wt) => transformed_states
                    Applied at the very end of the update step, if specified. Updates the attribute
                    `transformed_particles`. Useful when the particle state needs to be projected
                    into a different space.
        s_proportion : float
                    proportion of samples to draw from the initial on each iteration.
        n_eff_threshold=1.0: float
                    effective sample size at which resampling will be performed (0.0->1.0). Values
                    <1.0 will allow samples to propagate without the resampling step until
                    the effective sample size (n_eff) drops below the specified threshold.
        column_names : list of strings
                    names of each the columns of the state vector

        """
        self.s_fn = s_fn or s
        self.column_names = column_names
        self.prior_fn = prior_fn
        self.n_particles = n_particles
        self.init_filter()
        self.n_eff_threshold = n_eff_threshold
        self.d = self.particles.shape[1]
        self.observe_fn = observe_fn or unique_naming
        self.dynamics_fn = dynamics_fn or unique_naming
        self.noise_fn = noise_fn or unique_naming
        self.weight_fn = weight_fn or squared_error
        self.transform_fn = transform_fn
        self.transformed_particles = None
        self.s_proportion = s_proportion or 0.0
        self.internal_weight_fn = internal_weight_fn
        self.original_particles = numpy.array(self.particles)

    def init_filter(self, mask=None):
        """Initialise the filter by drawing samples from the prior.

        Parameters:
        -----------
        mask : array, optional
            boolean mask specifying the elements of the particle array to draw from the prior. None (default)
            implies all particles will be sd (i.e. a complete reset)
        """
        new_sample = self.prior_fn(self.n_particles)

        # s from the prior
        if mask is None:
            self.particles = new_sample
        else:
            self.particles[mask, :] = new_sample[mask, :]

    def update(self, observed=None, **kwargs):
        """Update the state of the particle filter given an observation.

        Parameters:
        ----------

        observed: array
            The observed output, in the same format as observe_fn() will produce. This is typically the
            inumpyut from the sensor observing the process (e.g. a camera image in optical tracking).
            If None, then the observation step is skipped, and the filter will run one step in prediction-only mode.

        kwargs: any keyword arguments specified will be passed on to:
            observe_fn(y, **kwargs)
            weight_fn(x, **kwargs)
            dynamics_fn(x, **kwargs)
            noise_fn(x, **kwargs)
            internal_weight_function(x, y, **kwargs)
            transform_fn(x, **kwargs)
        """

        # apply dynamics and noise
        self.particles = self.noise_fn(
            self.dynamics_fn(self.particles, **kwargs), **kwargs
        )

        # hypothesise observations
        self.hypotheses = self.observe_fn(self.particles, **kwargs)

        if observed is not None:
            # compute similarity to observations
            # force to be positive

            wt = numpy.clip(
                numpy.array(
                    self.weight_fn(
                        self.hypotheses.reshape(self.n_particles, -1),
                        observed.reshape(1, -1),
                        **kwargs
                    )
                ),
                0,
                numpy.inf,
            )
        else:
            # we have no observation, so all particles weighted the same
            wt = numpy.ones((self.n_particles,))

        # apply weighting based on the internal state
        # most filters don't use this, but can be a useful way of combining
        # forward and inverse models
        if self.internal_weight_fn is not None:
            internal_wt = self.internal_weight_fn(
                self.particles, observed, **kwargs
            )
            internal_wt = numpy.clip(internal_wt, 0, numpy.inf)
            internal_wt = internal_wt / numpy.tot(internal_wt)
            wt *= internal_wt

        # normalise wt to resampling probabilities
        #self.weight_normalisation = numpy.tot(wt)
        self.weight_normalisation = numpy.sum(wt)
        self.wt = wt / self.weight_normalisation

        # Compute effective sample size and entropy of weighting vector.
        # These are useful statistics for adaptive particle filtering.
        #self.n_eff = (1.0 / numpy.tot(self.wt ** 2)) / self.n_particles
        #self.weight_entropy = numpy.tot(self.wt * numpy.log(self.wt))
        self.n_eff = (1.0 / numpy.sum(self.wt ** 2)) / self.n_particles
        self.weight_entropy = numpy.sum(self.wt * numpy.log(self.wt))

        # preserve current sample set before any replenishment
        self.original_particles = numpy.array(self.particles)

        # store mean (expected) hypothesis
        #self.mean_hypothesis = numpy.tot(self.hypotheses.T * self.wt, axis=-1).T
        #self.mean_state = numpy.tot(self.particles.T * self.wt, axis=-1).T
        self.mean_hypothesis = numpy.sum(self.hypotheses.T * self.wt, axis=-1).T
        self.mean_state = numpy.sum(self.particles.T * self.wt, axis=-1).T
        self.cov_state = numpy.cov(self.particles, rowvar=False, aweights=self.wt)

        # store MAP estimate
        argmax_weight = numpy.argmax(self.wt)
        self.map_state = self.particles[argmax_weight]
        self.map_hypothesis = self.hypotheses[argmax_weight]

        # apply any post-processing
        if self.transform_fn:
            self.transformed_particles = self.transform_fn(
                self.original_particles, self.wt, **kwargs
            )
        else:
            self.transformed_particles = self.original_particles
        # randomly s some particles from the prior
        random_mask = (
            numpy.random.random(size=(self.n_particles,)) < self.s_proportion
        )

        # resampling (systematic resampling) step
        if self.n_eff < self.n_eff_threshold:
            indices = self.s_fn(self.wt)
            self.particles = self.particles[indices, :]
            # self.wt = self.wt[indices]

        self.sd_particles = random_mask
        self.init_filter(mask=random_mask)
