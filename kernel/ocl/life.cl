#include "kernel/ocl/common.cl"

__kernel void life_ocl (__global char *in, __global char *out)
{
  int x    = get_global_id (0);
  int y    = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  if (x == 0 || y == 0 || x == DIM - 1 || y == DIM - 1)
    return;

  unsigned me = in[y * DIM + x];
  unsigned n  = in[(y - 1) * DIM + x - 1]
              + in[(y - 1) * DIM + x]
              + in[(y - 1) * DIM + x + 1]
              + in[y * DIM + x - 1]
              + in[y * DIM + x + 1]
              + in[(y + 1) * DIM + x - 1]
              + in[(y + 1) * DIM + x]
              + in[(y + 1) * DIM + x + 1];

  out[y * DIM + x] = (n == 3) || (me && (n == 2));
}

__kernel void life_ocl_hybrid (__global char *in, __global char *out, unsigned startingY)
{
  int x    = get_global_id (0);
  int y    = get_global_id (1) + startingY;
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  if (x == 0 || y == 0 || x == DIM - 1 || y == DIM - 1) return;

  unsigned me, n;

    me = in[y * DIM + x];
    n  = in[(y - 1) * DIM + x - 1]
       + in[(y - 1) * DIM + x]
       + in[(y - 1) * DIM + x + 1]
       + in[y * DIM + x - 1]
       + in[y * DIM + x + 1]
       + in[(y + 1) * DIM + x - 1]
       + in[(y + 1) * DIM + x]
       + in[(y + 1) * DIM + x + 1];

    out[y * DIM + x] = (n == 3) || (me && (n == 2));
}

__kernel void life_update_texture (__global char *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  unsigned color = cur [y * DIM + x] * rgb (255, 255, 0);

  write_imagef (tex, (int2)(x, y), color_to_float4 (color));
}