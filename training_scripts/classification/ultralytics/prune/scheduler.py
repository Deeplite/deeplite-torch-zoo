from abc import ABC, abstractmethod


class _Scheduler(ABC):
    def __init__(self, begin_step, frequency, dtype):
        self.begin_step = begin_step
        self.frequency = frequency
        self.dtype = dtype

    @staticmethod
    def _validate_step(begin_step, frequency, allow_negative_1):
        """Checks whether the parameters for pruning schedule are valid.
        Args:
            begin_step: Step at which to begin pruning.
            frequency: Only apply pruning every `frequency` steps.
            allow_negative_1: Whether end_step is allowed to be `-1` or not.
        Returns:
            None
        """
        if begin_step < 0:
            raise ValueError('begin_step should be >= 0')
        # In cases like PolynomialDecay, continuing to prune forever does not
        # make sense. The function needs an end_step to decay the sparsity.
        if frequency <= 0:
            raise ValueError('frequency should be > 0')

    @staticmethod
    def _validate_value(value, variable_name):
        if not 0.0 <= value:
            raise ValueError('{} must be in range [0,1)'.format(variable_name))

    @classmethod
    def from_config(cls, config):
        """Instantiates a `PruningSchedule` from its config.
        Args:
                config: Output of `get_config()`.
        Returns:
                A `PruningSchedule` instance.
        """
        return cls(**config)

    @abstractmethod
    def get_config(self):
        raise NotImplementedError(
            'PruningSchedule implementation override get_config'
        )

    @abstractmethod
    def _get_value(self, step):
        pass

    def get_value(self, step):
        if step <= self.begin_step:
            step += 1
            value = self.initial_value
        else:
            value = self._get_value(step)
        if self.dtype == 'int':
            return int(value)
        return value

    def should_do(self, step):
        """Checks if action should be applied in the current training step.
        Pruning should only occur within the [`begin_step`, `end_step`] range
        every `frequency` number of steps.
        Args:
            step: Current training step.
            begin_step: Step at which to begin pruning.
            end_step: Step at which to end pruning.
            frequency: Only apply pruning every `frequency` steps.
        Returns:
            True/False, if pruning should be applied in current step.
        """
        is_in_action_range = (step + 1) >= self.begin_step
        is_action_turn = ((step - self.begin_step) % self.frequency) == 0
        return is_in_action_range and is_action_turn


class ConstantScheduler(_Scheduler):
    """Schedule with constant value throughout training."""

    def __init__(self, begin_step, frequency, dtype, initial_value):
        super().__init__(begin_step, frequency, dtype)
        self.end_step = float('inf')
        self.initial_value = initial_value
        self._validate_step(self.begin_step, self.frequency, True)
        self._validate_value(initial_value, 'initial_value')

    def _get_value(self, step):
        return self.initial_value

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'begin_step': self.begin_step,
                'frequency': self.frequency,
                'initial_value': self.initial_value,
            },
        }


class LinearScheduler(_Scheduler):
    """Schedule with linear increasement throughout training."""

    def __init__(self, begin_step, frequency, dtype, initial_value, gain):
        """Initializes a Pruning schedule with linear slope sparsity.
        Sparsity is applied in the interval [`begin_step`, `INF+'] every
        `frequency` steps. At each applicable step, the value(%) is computed
        according the linear equation:
            value = slope * step + initial_value.
        NB: If slope = 0 the value(%) is always constant.
        Args:
            begin_step: Step at which to begin pruning.
            frequency: Only apply every `frequency` steps.
            initial_valuey: A scalar float representing the initial value.
            gain: A scalar positive float representing the linear gain.
        """
        super().__init__(begin_step, frequency, dtype)
        self.end_step = float('inf')
        self.initial_value = initial_value
        self.gain = gain
        self.slope = self.gain / self.frequency
        self._validate_step(self.begin_step, self.frequency, True)
        self._validate_value(initial_value, 'initial_value')

    def _get_value(self, step):
        return self.initial_value + self.slope * (step - self.begin_step)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'begin_step': self.begin_step,
                'frequency': self.frequency,
                'initial_value': self.initial_value,
                'gain': self.gain,
            },
        }


class PolyDecayScheduler(_Scheduler):
    """Schedule with polyomial-decay function."""

    def __init__(
        self,
        begin_step,
        frequency,
        dtype,
        initial_value,
        final_value,
        end_step,
        power,
    ):
        """Initializes a Pruning schedule with a PolynomialDecay function.
        Pruning rate grows rapidly in the beginning from initial_sparsity, but
        then  plateaus slowly to the target sparsity. The function applied is
        value = final_value + (initial_value - final_value)
            * (1 - (step - begin_step)/(end_step - begin_step)) ^ exponent
        which is a polynomial decay function.
        See [paper](https://arxiv.org/abs/1710.01878).
        Args:
            initial_value: Value at which scheduled action begins.
            final_value: Value at which scheduled action ends.
            begin_step: Step at which to begin scheduled action.
            end_step: Step at which to end scheduled action.
            power: Exponent to be used in the poly-decay function.
            frequency: Only apply scheduled action every `frequency` steps.
       """
        super().__init__(begin_step, frequency, dtype)
        self.initial_value = initial_value
        self.final_value = final_value
        self.end_step = end_step
        self.power = power
        self._validate_step(self.begin_step, self.frequency, True)
        self._validate_value(initial_value, 'initial_value')

    def _get_value(self, step):
        p = min(
            1.0,
            max(
                0.0,
                (step - self.begin_step) / ((self.end_step - self.begin_step)),
            ),
        )
        return self.final_value + (self.initial_value - self.final_value) * (
            (1 - p) ** self.power
        )

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'begin_step': self.begin_step,
                'end_step': self.end_step,
                'frequency': self.frequency,
                'initial_value': self.initial_value,
                'final_value': self.final_value,
                'power': self.power,
            },
        }
