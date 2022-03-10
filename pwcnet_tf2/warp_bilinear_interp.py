import imp
import os, sys
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops


def get_grid(x):
    batch_size, H, W, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(H), tf.range(W),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11

def _interpolate_bilinear(grid, query_points, name='interpolate_bilinear', indexing='ij'):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
      name: a name for the operation (optional).
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`
    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')
    
    # batch_q_pts_on_grid_flattened = array_ops.reshape(batch_q_pts_on_grid, [B, H * W, 2])
    # interpolated = _interpolate_bilinear(image, batch_q_pts_on_grid_flattened)

    with ops.name_scope(name):
        grid = ops.convert_to_tensor(grid)
        query_points = ops.convert_to_tensor(query_points)
        shape = array_ops.unstack(array_ops.shape(grid))
        if len(shape) != 4:
            msg = 'Grid must be 4 dimensional. Received: '
            raise ValueError(msg + str(shape))

        batch_size, height, width, channels = shape
        query_type = query_points.dtype
        query_shape = array_ops.unstack(array_ops.shape(query_points))  # (B, H*W, 2)
        grid_type = grid.dtype

        if len(query_shape) != 3:
            msg = ('Query points must be 3 dimensional. Received: ')
            raise ValueError(msg + str(query_shape))

        _, num_queries, _ = query_shape

        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1] if indexing == 'ij' else [1, 0]
        unstacked_query_points = array_ops.unstack(query_points, axis=2)

        for dim in index_order:
            with ops.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]  # (B, H*W)

                size_in_indexing_dimension = shape[dim + 1]  # shape=(B, H, W, C)  -> (B, *, *, C)

                # ------------------------------------------------
                size_dim = shape[dim + 1]
                coord_min = constant_op.constant(0.0, dtype=query_type)
                coord_max = constant_op.constant(size_dim - 1, dtype=query_type)
                queries = math_ops.minimum(math_ops.maximum(coord_min, queries), coord_max)
                q_floor = math_ops.floor(queries)
                q_ceil = q_floor + 1
                alpha = math_ops.cast(queries - floor, grid_type)  # delta_x/y in [0, 1]
                # ------------------------------------------------

                # 查询点大小限制到 [0, H/W-1]
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)  # W/H - 2
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)  # delta_x/y
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)
                alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)

        flattened_grid = array_ops.reshape(grid,
                                           [batch_size * height * width, channels])
        batch_offsets = array_ops.reshape(
            math_ops.range(batch_size) * height * width, [batch_size, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords, name):
            with ops.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left')
        top_right = gather(floors[0], ceils[1], 'top_right')
        bottom_left = gather(ceils[0], floors[1], 'bottom_left')
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

        v11 = gather(floors[0], floors[1], 'top_left') # v_{yx}
        v12 = gather(floors[0], ceils[1], 'top_right')
        v21 = gather(ceils[0], floors[1], 'bottom_left')
        v22 = gather(ceils[0], ceils[1], 'bottom_right')

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp

def dense_image_warp(image, flow, name='dense_image_warp'):
    """Image warping using per-pixel flow vectors.

    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the  source image. Specifically, the
    pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.


    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).

    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.
    """
    batch_size, height, width, channels = array_ops.unstack(array_ops.shape(image))
    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = array_ops.meshgrid(
        math_ops.range(width), math_ops.range(height))
    stacked_grid = math_ops.cast(
        array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                                [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = _interpolate_bilinear(image, query_points_flattened)
    interpolated = array_ops.reshape(interpolated,
                                        [batch_size, height, width, channels])
    return interpolated


B = 4
H = 3
W = 5
C = 2

flow_x = tf.ones((B, H, W), dtype=tf.float32)
flow_y = 2* tf.ones((B, H, W), dtype=tf.float32)
flow = tf.stack([flow_x, flow_y], axis=3)
print(flow.shape)

print('-'*50)

grid_x, grid_y = array_ops.meshgrid(math_ops.range(W), math_ops.range(H))
stack_grid = math_ops.cast(array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
batch_grid = array_ops.expand_dims(stack_grid, axis=0)
batch_q_pts_on_grid = batch_grid - flow
# print(flow)
print('-'*50)
print(batch_grid)
print('-'*50)
# print(batch_q_pts_on_grid)
batch_q_pts_on_grid_flattened = array_ops.reshape(batch_q_pts_on_grid, [B, H * W, 2])

# interpolated = _interpolate_bilinear(image, query_points_flattened)
# interpolated = array_ops.reshape(interpolated,
#                                         [batch_size, height, width, channels])

t = tf.ones((H, W, 2), dtype=tf.float32)
unstacked_query_points = array_ops.unstack(t, axis=2)
print(unstacked_query_points)