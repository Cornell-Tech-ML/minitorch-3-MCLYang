import random
from .operators import prod
from numpy import array, float64, ndarray
import numba

# import pdb

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


def index_to_position(index, strides):
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index (array-like): index tuple of ints
        strides (array-like): tensor strides

    Returns:
        int : position in storage
    """

    position = 0
    for ind, strides in zip(index, strides):
        position += ind * strides
    return position

    # for i in range(len(index)):
    #     position = position + strides[i] * index[i]

    # for i in index:
    #     for s in strides:
    #         position = i*s+position

    # return position

    # TODO: Implement for Task 2.1.
    # raise NotImplementedError('Need to implement for Task 2.1')


def count(position, shape, out_index):
    """
    Convert a `position` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        position (int): current position.
        shape (tuple): tensor shape.
        out_index (array): the index corresponding to position.

    Returns:
      None : Fills in `out_index`.

    """
    # print('position == 0',position == 0)
    cur_pos = position + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_pos % sh)
        cur_pos = cur_pos // sh

    # TODO: Implement for Task 2.1.

    # raise NotImplementedError('Need to implement for Task 2.1')


def implicit_adding_one(shape1, shape2):
    bigger_shape = shape1
    smaller_shape = shape2
    if len(shape1) > len(shape2):
        bigger_shape = shape1
        smaller_shape = shape2
    else:
        bigger_shape = shape2
        smaller_shape = shape1

    num_add_ones = len(bigger_shape) - len(smaller_shape)
    new_smaller_shape = ()
    for i in range(num_add_ones):
        new_smaller_shape = new_smaller_shape + (1,)
    for i in range(len(smaller_shape)):
        new_smaller_shape = new_smaller_shape + (smaller_shape[i],)

    smaller_shape = new_smaller_shape
    return bigger_shape, smaller_shape, num_add_ones


def broadcast_index(big_index, big_shape, shape, out_index):
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index (array-like): multidimensional index of bigger tensor
        big_shape (array-like): tensor shape of bigger tensor
        shape (array-like): tensor shape of smaller tensor
        out_index (array-like): multidimensional index of smaller tensor

    Returns:
        None : Fills in `out_index`.
    """
    # compare_notes = True
    # if(len(big_shape) == len(shape)):
    #     for i,s in enumerate(shape):
    #         if shape[i] != big_shape[i]:
    #             compare_notes = False

    # else:
    #     compare_notes = False

    # if compare_notes:
    #     for i,s in enumerate(shape):
    #         if s>1:
    #             out_index[i] = big_index[i]
    #         else:
    #             out_index[i] = 0
    # else:
    #     for i,s in enumerate(shape):
    #         if s>1:
    #             out_index[i] = big_index[i+(len(big_shape)- len(shape))]
    #         else:
    #             out_index[i] = 0
    # return None

    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    return None


def shape_broadcast(shape1, shape2):
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 (tuple) : first shape
        shape2 (tuple) : second shape

    Returns:
        tuple : broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    a, b = shape1, shape2
    m = max(len(a), len(b))
    # print("m",m)
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError("Broadcast failure")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError("Broadcast failure")
    return tuple(reversed(c_rev))

    # shape1 = tuple(shape1)
    # shape2 = tuple(shape2)
    # # print("shape1",shape1)
    # # print("shape2",shape2)
    # # print()
    # if shape1 == shape2:
    #     return shape1
    # elif shape1 == (1,) or shape2 == (1,):
    #     if shape1 == (1,):
    #         return shape2
    #     else:
    #         return shape1
    # else:
    #     bigger_shape, smaller_shape = shape1, shape2
    #     if len(shape1) != len(shape2):
    #         bigger_shape, smaller_shape, _ = implicit_adding_one(shape1, shape2)

    #     assert len(bigger_shape) == len(
    #         smaller_shape
    #     ), "two len(shape) are not same after fix"

    #     # only 1 dimention is different and one of them for that dim is 1
    #     new_shape = tuple()

    #     for i, v in enumerate(bigger_shape):
    #         greater_num = 0
    #         less_num = 0

    #         if bigger_shape[i] > smaller_shape[i]:
    #             greater_num = bigger_shape[i]
    #             less_num = smaller_shape[i]
    #         elif bigger_shape[i] < smaller_shape[i]:
    #             greater_num = smaller_shape[i]
    #             less_num = bigger_shape[i]

    #         elif bigger_shape[i] == smaller_shape[i]:
    #             greater_num = bigger_shape[i]
    #             less_num = bigger_shape[i]

    #         if greater_num == less_num:
    #             new_shape = new_shape + (greater_num,)
    #         else:
    #             # print("shape1",shape1)
    #             # print("shape2",shape2)
    #             # print("bigger_shape[i]",bigger_shape[i])
    #             # print("smaller_shape[i]",smaller_shape[i])
    #             # print()
    #             # assert (less_num == 1), "Boardcoasting failed, check tensor shape"
    #             new_shape = new_shape + (greater_num,)

    #     return new_shape


def strides_from_shape(shape):
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    def __init__(self, storage, shape, strides=None):
        if isinstance(storage, ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        self._storage = self._storage

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"

        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self):
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self):
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a, shape_b):
        return shape_broadcast(shape_a, shape_b)

    def index(self, index):
        if isinstance(index, int):
            index = array([index])
        if isinstance(index, tuple):
            index = array(index)

        # Check for errors
        if index.shape[0] != len(self.shape):
            raise IndexingError(f"Index {index} must be size of {self.shape}.")
        for i, ind in enumerate(index):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {index} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {index} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self):
        lshape = array(self.shape)
        out_index = array(self.shape)
        for i in range(self.size):
            count(i, lshape, out_index)
            # print('i',i)
            # print('lshape',lshape)
            # print('out_index',out_index)
            # print()
            yield tuple(out_index)

    def sample(self):
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key):
        return self._storage[self.index(key)]

    def set(self, key, val):
        self._storage[self.index(key)] = val

    def tuple(self):
        return (self._storage, self._shape, self._strides)

    def permute(self, *order):
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            :class:`TensorData`: a new TensorData with the same storage and a new dimension order.
        """
        # pdb.set_trace()
        # print(order)
        # print(self.shape)
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        # pdb.set_trace()
        # if(list(order) == list(range(len(self.shape))):
        # print('order',order)
        # print('list(sorted(order))',list(sorted(order)))
        # print('self.shape',self.shape)
        # print('list(range(len(self.shape))',list(range(len(self.shape))))
        # print('list(order)',list(order))
        # print(1)
        # if(   list(order) == list(range(len(self.shape)))   ):
        #     newtd= TensorData(self._storage, self.shape, strides=self.strides)

        # else:
        shape = tuple()
        strides = tuple()
        for i, p in enumerate(order):
            shape = shape + (self.shape[p],)
            strides = strides + (self._strides[p],)

        newtd = TensorData(self._storage, shape=shape, strides=strides)
        return newtd

        # TODO: Implement for Task 2.1.
        # raise NotImplementedError('Need to implement for Task 2.1')

    def to_string(self):
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
