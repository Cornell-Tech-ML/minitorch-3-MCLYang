from .autodiff import FunctionBase, Variable, History
from . import operators
import numpy as np

# import pdb

## Task 1.1
## Derivatives


def central_difference(f, *vals, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
       f : arbitrary function from n-scalar args to one value
       *vals (floats): n-float values :math:`x_0 \ldots x_{n-1}`
       arg (int): the number :math:`i` of the arg to compute the derivative
       epsilon (float): a small constant

    Returns:
       float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    # TODO: Implement for Task 1.1.
    x_i = vals[arg]
    x_i_next = x_i + epsilon
    x_i_previous = x_i - epsilon

    vals_1 = []
    vals_0 = []
    for i in range(len(vals)):
        if i == arg:
            vals_1.append(x_i_next)
            vals_0.append(x_i_previous)
        else:
            vals_1.append(vals[i])
            vals_0.append(vals[i])

    vals_1 = tuple(vals_1)
    vals_0 = tuple(vals_0)

    f1 = f(*vals_1)
    f0 = f(*vals_0)

    f_prime = (f1 - f0) / (2 * epsilon)
    # pdb.set_trace()

    return f_prime

    # raise NotImplementedError('Need to implement for Task 1.1')


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
        self.data = v

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __add__(self, b):
        # TODO: Implement for Task 1.2.
        return Add.apply(self, b)

    #        raise NotImplementedError('Need to implement for Task 1.2')

    def __lt__(self, b):
        return LT.apply(self, b)
        # TODO: Implement for Task 1.2.

    #       raise NotImplementedError('Need to implement for Task 1.2')

    def __gt__(self, b):
        return LT.apply(b, self)
        # TODO: Implement for Task 1.2.
        #      raise NotImplementedError('Need to implement for Task 1.2')

    def __sub__(self, b):
        return Add.apply(self, -b)
        # TODO: Implement for Task 1.2.

    #     raise NotImplementedError('Need to implement for Task 1.2')

    def __neg__(self):
        return Neg.apply(self)
        # TODO: Implement for Task 1.2.

    #    raise NotImplementedError('Need to implement for Task 1.2')

    def log(self):
        return Log.apply(self)
        # TODO: Implement for Task 1.2.

    #    raise NotImplementedError('Need to implement for Task 1.2')

    def exp(self):
        return Exp.apply(self)
        # TODO: Implement for Task 1.2.

    #    raise NotImplementedError('Need to implement for Task 1.2')

    def sigmoid(self):
        return Sigmoid.apply(self)
        # TODO: Implement for Task 1.2.

    #    raise NotImplementedError('Need to implement for Task 1.2')

    def relu(self):
        return ReLU.apply(self)
        # TODO: Implement for Task 1.2.

    #    raise NotImplementedError('Need to implement for Task 1.2')

    def get_data(self):
        return self.data


class ScalarFunction(FunctionBase):
    "A function that processes and produces Scalar variables."

    @staticmethod
    def forward(ctx, *inputs):
        """Args:

           ctx (:class:`Context`): A special container object to save
                                   any information that may be needed for the call to backward.
           *inputs (list of numbers): Numerical arguments.

        Returns:
            number : The computation of the function :math:`f`

        """
        pass

    @staticmethod
    def backward(ctx, d_out):
        """
        Args:
            ctx (Context): A special container object holding any information saved during in the corresponding `forward` call.
            d_out (number):
        Returns:
            numbers : The computation of the derivative function :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.
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
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward((a, b))

        return operators.mul(a, b)

        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        x_prime, y_prime = ctx.saved_values

        return (y_prime * d_output, x_prime * d_output)
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)

        return operators.inv(a)
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        x_prime = ctx.saved_values

        x_prime = -1 / (x_prime) ** 2
        return x_prime * d_output

        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        return operators.neg(a)

        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        sigma = ctx.saved_values
        return sigma * (1.0 - sigma) * d_output
        # return (
        #     d_output * (operators.sigmoid(x_prime)) * (1 - operators.sigmoid(x_prime))
        # )

        # raise NotImplementedError('Need to implement for Task 1.4')


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)

        return operators.relu(a)

        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        x_prime = ctx.saved_values
        if x_prime <= 0:
            return 0
        else:
            return d_output

        # TODO: Implement for Task 1.4.

        # raise NotImplementedError('Need to implement for Task 1.4')


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)

        return operators.exp(a)

        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        x_prime = ctx.saved_values

        return operators.exp(x_prime) * d_output
        # raise NotImplementedError('Need to implement for Task 1.4')


def derivative_check(f, *scalars):

    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    # import pdb
    # pdb.set_trace()
    out.backward()

    vals = [v for v in scalars]

    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        print("x.derivative", x.derivative)
        print("check.data", check.data)
        np.testing.assert_allclose(x.derivative, check.data, 1e-2, 1e-2)
