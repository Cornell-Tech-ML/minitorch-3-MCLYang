from .autodiff import FunctionBase, Variable, History
from . import operators
import numpy as np


## Task 1.1
## Derivatives


def central_difference(f, *vals, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`
        arg (int): the number :math:`i` of the arg to compute the derivative
        epsilon (float): a small constant

    Returns:
        float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    raise NotImplementedError('Need to include this file from past assignment.')


## Task 1.2 and 1.4
## Scalar Forward and Backward


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.

    """

    def __init__(self, v, back=History(), name=None):
        super().__init__(back, name=name)
        self.data = float(v)

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __add__(self, b):
        raise NotImplementedError('Need to include this file from past assignment.')

    def __lt__(self, b):
        raise NotImplementedError('Need to include this file from past assignment.')

    def __gt__(self, b):
        raise NotImplementedError('Need to include this file from past assignment.')

    def __sub__(self, b):
        raise NotImplementedError('Need to include this file from past assignment.')

    def __neg__(self):
        raise NotImplementedError('Need to include this file from past assignment.')

    def log(self):
        raise NotImplementedError('Need to include this file from past assignment.')

    def exp(self):
        raise NotImplementedError('Need to include this file from past assignment.')

    def sigmoid(self):
        raise NotImplementedError('Need to include this file from past assignment.')

    def relu(self):
        raise NotImplementedError('Need to include this file from past assignment.')

    def get_data(self):
        return self.data


class ScalarFunction(FunctionBase):
    "A function that processes and produces Scalar variables."

    @staticmethod
    def forward(ctx, *inputs):
        """
        Forward call, compute f(`inputs`)

        Args:
            ctx (:class:`Context`): A special container object to save
                                    any information that may be needed for the call to backward.
            *inputs (list of floats): Numerical arguments.

        Should return float the computation of the function :math:`f`
        """
        pass

    @staticmethod
    def backward(ctx, d_out):
        """
        Backward call, compute f'_{x_i}(`inputs`) * d_out

        Args:
            ctx (Context): A special container object holding any information saved during in the corresponding `forward` call.
            d_out (float): :math:`d_out` term in chain rule.

        Should return the computation of the derivative function :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.
        """
        pass

    # checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


# Examples
class Add(ScalarFunction):
    "Addition function"

    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, d_output):
        return d_output, d_output


class Log(ScalarFunction):
    "Log function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.log_back(a, d_output)


class LT(ScalarFunction):
    "Less-than function"

    @staticmethod
    def forward(ctx, a, b):
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx, d_output):
        return 0.0, 0.0


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx, a, b):
        raise NotImplementedError('Need to include this file from past assignment.')

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError('Need to include this file from past assignment.')


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError('Need to include this file from past assignment.')

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError('Need to include this file from past assignment.')


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError('Need to include this file from past assignment.')

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError('Need to include this file from past assignment.')


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError('Need to include this file from past assignment.')

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError('Need to include this file from past assignment.')


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError('Need to include this file from past assignment.')

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError('Need to include this file from past assignment.')


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError('Need to include this file from past assignment.')

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError('Need to include this file from past assignment.')


def derivative_check(f, *scalars):

    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    out.backward()

    vals = [v for v in scalars]

    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        print(x.derivative, check)
        np.testing.assert_allclose(x.derivative, check.data, 1e-2, 1e-2)
