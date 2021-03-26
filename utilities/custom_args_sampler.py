from dimod import ComposedSampler, Sampler


class CustomArgsSampler(ComposedSampler):
    """Class to submit custom sampling kwargs to dimod sampler :class:`.Sampler`.

    See issue on GitHub Problem Label Feature with dwave-qiskit-plugin:
        `randomir <https://github.com/dwavesystems/dwave-qiskit-plugin/issues/1>`
    """

    def __init__(self, child_sampler: Sampler, sample_kwargs=None):
        self._children = [child_sampler]
        if sample_kwargs is None:
            sample_kwargs = {}
        self._sample_kwargs = sample_kwargs

    @property
    def children(self):
        """list[ :obj:`.Sampler`]: List of child samplers that are used by
        this composite.
        """
        return self._children

    @property
    def parameters(self):
        """dict: A dict where keys are the keyword parameters accepted by the sampler
        methods and values are lists of the properties relevent to each parameter.
        """
        return self.child.parameters.copy()

    @property
    def properties(self):
        """dict: A dict containing any additional information about the sampler.
        """
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, **kwargs):
        """Sample from a binary quadratic model by invoking sampling method with custom args.

        Args:
            :obj:`.BinaryQuadraticModel`:
                A binary quadratic model.

            **kwargs:
                See the implemented sampling for additional keyword definitions.

        Returns:
            :obj:`.SampleSet`

        """
        kwargs.update(self._sample_kwargs)
        return self.child.sample(bqm, **kwargs)

    def set_label(self, label: str) -> None:
        """Sets the label for future problem submissions.

        Args:
            label (str): Problem label to be used in :meth:`.sample`.
        """
        self._sample_kwargs['label'] = label
