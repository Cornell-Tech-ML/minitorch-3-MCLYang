# 3.1 analysis[see /project/analysis.txt]
(5781) (base) malcolm@Malcolm:/media/malcolm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang$ python project/parallel_test.py
MAP
/media/malcolm/1E577EB53AA8D6D4/cornell_class/5781/5781/lib/python3.7/site-packages/numba/np/ufunc/parallel.py:363: NumbaWarning: The TBB threading layer requires TBB version 2019.5 or later i.e., TBB_INTERFACE_VERSION >= 11005. Found TBB_INTERFACE_VERSION = 9002. The TBB threading layer is disabled.
  warnings.warn(problem)

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /media/mal
colm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.
py (67)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /media/malcolm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (67)
--------------------------------------------------------------------------------|loop #ID
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):    |
        # out_index = np.zeros(MAX_DIMS,np.int32)                               |
        # in_index = np.zeros(MAX_DIMS,np.int32)                                |
                                                                                |
        for i in prange(len(out)):----------------------------------------------| #2
            out_index = np.zeros(MAX_DIMS,np.int32)-----------------------------| #0
            in_index = np.zeros(MAX_DIMS,np.int32)------------------------------| #1
            count(i,out_shape,out_index)                                        |
            broadcast_index(out_index,out_shape,in_shape,in_index)              |
            o = index_to_position(out_index,out_strides)                        |
            j = index_to_position(in_index,in_strides)                          |
            out[o] = fn(in_storage[j])                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #2) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#2).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (72) is
hoisted out of the parallel loop labelled #2 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (73) is
hoisted out of the parallel loop labelled #2 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /media/mal
colm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.
py (137)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /media/malcolm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (137)
--------------------------------------------------------------------|loop #ID
    def _zip(                                                       |
        out,                                                        |
        out_shape,                                                  |
        out_strides,                                                |
        a_storage,                                                  |
        a_shape,                                                    |
        a_strides,                                                  |
        b_storage,                                                  |
        b_shape,                                                    |
        b_strides,                                                  |
    ):                                                              |
                                                                    |
                                                                    |
        for i in prange(len(out)):----------------------------------| #6
            out_index = np.zeros(MAX_DIMS,np.int32)-----------------| #3
            a_index = np.zeros(MAX_DIMS,np.int32)-------------------| #4
            b_index = np.zeros(MAX_DIMS,np.int32)-------------------| #5
            count(i,out_shape,out_index)                            |
            o = index_to_position(out_index,out_strides)            |
            broadcast_index(out_index,out_shape,a_shape,a_index)    |
            j = index_to_position(a_index,a_strides)                |
            broadcast_index(out_index,out_shape,b_shape,b_index)    |
            k = index_to_position(b_index,b_strides)                |
            out[o] = fn(a_storage[j],b_storage[k])                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #6, #3, #4, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--6 is a parallel loop
   +--3 --> rewritten as a serial loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (serial)
   +--4 (serial)
   +--5 (serial)



Parallel region 0 (loop #6) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (151) is
hoisted out of the parallel loop labelled #6 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (152) is
hoisted out of the parallel loop labelled #6 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (153) is
hoisted out of the parallel loop labelled #6 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /med
ia/malcolm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fas
t_ops.py (211)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /media/malcolm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (211)
--------------------------------------------------------------|loop #ID
    def _reduce(                                              |
        out,                                                  |
        out_shape,                                            |
        out_strides,                                          |
        a_storage,                                            |
        a_shape,                                              |
        a_strides,                                            |
        reduce_shape,                                         |
        reduce_size,                                          |
    ):                                                        |
                                                              |
                                                              |
        for i in prange(len(out)):----------------------------| #9
            out_index = np.zeros(MAX_DIMS,np.int32)-----------| #7
            count(i,out_shape,out_index)                      |
            o = index_to_position(out_index,out_strides)      |
            for s in range(reduce_size):                      |
                a_index = np.zeros(MAX_DIMS,np.int32)---------| #8
                count(s,reduce_shape,a_index)                 |
                for n in range(len(reduce_shape)):            |
                    if reduce_shape[n]!=1:                    |
                        out_index[n] = a_index[n]             |
                                                              |
                j = index_to_position(out_index,a_strides)    |
                out[o] = fn(out[o],a_storage[j])              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #9, #7, #8).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--9 is a parallel loop
   +--8 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (serial)
   +--7 (serial)



Parallel region 0 (loop #9) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#9).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (224) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (228) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_matrix_multiply, /media/malco
lm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py
 (297)
================================================================================


Parallel loop listing for  Function tensor_matrix_multiply, /media/malcolm/1E577EB53AA8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (297)
----------------------------------------------------------------------|loop #ID
@njit(parallel=True)                                                  |
def tensor_matrix_multiply(                                           |
    out,                                                              |
    out_shape,                                                        |
    out_strides,                                                      |
    a_storage,                                                        |
    a_shape,                                                          |
    a_strides,                                                        |
    b_storage,                                                        |
    b_shape,                                                          |
    b_strides,                                                        |
):                                                                    |
    """                                                               |
    NUMBA tensor matrix multiply function.                            |
                                                                      |
    Should work for any tensor shapes that broadcast as long as ::    |
                                                                      |
        assert a_shape[-1] == b_shape[-2]                             |
                                                                      |
    Args:                                                             |
        out (array): storage for `out` tensor                         |
        out_shape (array): shape for `out` tensor                     |
        out_strides (array): strides for `out` tensor                 |
        a_storage (array): storage for `a` tensor                     |
        a_shape (array): shape for `a` tensor                         |
        a_strides (array): strides for `a` tensor                     |
        b_storage (array): storage for `b` tensor                     |
        b_shape (array): shape for `b` tensor                         |
        b_strides (array): strides for `b` tensor                     |
                                                                      |
    Returns:                                                          |
        None : Fills in `out`                                         |
    """                                                               |
    # out,                                                            |
    # out_shape,                                                      |
    # out_strides,                                                    |
    # print("a")                                                      |
    # for mm in a_shape:                                              |
    #     print(mm)                                                   |
                                                                      |
    # print("b")                                                      |
    # for mm in b_shape:                                              |
    #     print(mm)                                                   |
    # print("a")                                                      |
    # print(len(out_shape))                                           |
                                                                      |
    iteration_n = a_shape[-1]                                         |
                                                                      |
    for i in prange(len(out)):----------------------------------------| #12
        out_index = np.zeros(MAX_DIMS,np.int32)-----------------------| #10
        count(i,out_shape,out_index)                                  |
        o = index_to_position(out_index,out_strides)                  |
        a_index = np.copy(out_index)                                  |
        b_index = np.zeros(MAX_DIMS,np.int32)-------------------------| #11
        a_index[len(out_shape)-1] = 0                                 |
        b_index[len(out_shape)-2] = 0                                 |
        b_index[len(out_shape)-1] = out_index[len(out_shape)-1]       |
        temp_sum = 0                                                  |
        for w in range(iteration_n):                                  |
            # a_index = [d,a_row,w]                                   |
            # b_index = [0,w,b_col]                                   |
            a_index[len(out_shape)-1] = w                             |
            b_index[len(out_shape)-2] = w                             |
                                                                      |
            j = index_to_position(a_index,a_strides)                  |
            m = index_to_position(b_index,b_strides)                  |
            temp_sum = temp_sum + a_storage[j]*b_storage[m]           |
                                                                      |
        out[o] = temp_sum                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #12, #11, #10).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--12 is a parallel loop
   +--10 --> rewritten as a serial loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--10 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--10 (serial)
   +--11 (serial)



Parallel region 0 (loop #12) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#12).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (350) is
hoisted out of the parallel loop labelled #12 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /media/malcolm/1E577EB53AA
8D6D4/cornell_class/5781/minitorch-3-MCLYang/minitorch/fast_ops.py (346) is
hoisted out of the parallel loop labelled #12 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS,np.int32)
    - numpy.empty() is used for the allocation.
None
(5781) (base) malcolm@Malcolm:/media/malcolm/1E577

# 3.5
# simple 
![alt text](https://github.com/Cornell-Tech-ML/minitorch-3-MCLYang/blob/master/project/simple_graph.PNG)
# split 
![alt text](https://github.com/Cornell-Tech-ML/minitorch-3-MCLYang/blob/master/project/split_graph.PNG)
# xor 
![alt text](https://github.com/Cornell-Tech-ML/minitorch-3-MCLYang/blob/master/project/xor_graph.PNG)
# GPU VS CPU
# CPU
![alt text](https://github.com/Cornell-Tech-ML/minitorch-3-MCLYang/blob/master/project/biggermodel_cpu_graph.PNG)
![alt text](https://github.com/Cornell-Tech-ML/minitorch-3-MCLYang/blob/master/project/biggermodel_cpu_time.PNG)
# GPU
![alt text](https://github.com/Cornell-Tech-ML/minitorch-3-MCLYang/blob/master/project/biggermodel_gpu_graph.PNG)
![alt text](https://github.com/Cornell-Tech-ML/minitorch-3-MCLYang/blob/master/project/biggermodel_gpu_time.PNG)




[![Work in Repl.it](https://classroom.github.com/assets/work-in-replit-14baed9a392b3a25080506f3b7b6d57f295ec2978f6f33ec97e36a161684cbe9.svg)](https://classroom.github.com/online_ide?assignment_repo_id=3552982&assignment_repo_type=AssignmentRepo)
# MiniTorch Module 3

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html

This module requires `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 2.

You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.
