#include "easypap.h"
#include "rle_lexer.h"

#include <ezm.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define LIFE_COLOR (ezv_rgb (255, 255, 0))

typedef char cell_t;

static cell_t *_table = NULL, *_alternate_table = NULL;
static bool *active= NULL;
static bool *new_active= NULL;
#define TILES_X (DIM / TILE_W)
#define TILES_Y (DIM / TILE_H)
#define LINE_SIZE (DIM * sizeof (cell_t))

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

#define table_tile(table, y, x) (table)[(y) * TILES_X + (x)]

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

void life_init (void)
{
  // life_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    const unsigned size = DIM * LINE_SIZE;

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }

  if(active==NULL){
    active = mmap(NULL, (TILES_X) * (TILES_Y) * sizeof(bool),
                  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    new_active = mmap(NULL, (TILES_X) * (TILES_Y) * sizeof(bool),
                  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (int i = 0; i < TILES_Y; i++)
      for (int j = 0; j < TILES_X; j++)
        table_tile(active, i, j) = true;
  }
}

void life_finalize (void)
{
  const unsigned size = DIM * LINE_SIZE;

  munmap (_table, size);
  munmap (_alternate_table, size);
  munmap (active, (DIM / TILE_W) * (DIM / TILE_H) * sizeof(bool));
  munmap (new_active, (DIM / TILE_W) * (DIM / TILE_H) * sizeof(bool));
}

void life_config_ocl_hybrid (char *param)
{
  // Force data to be synch'ed on host for display
  easypap_gl_buffer_sharing = 0;
}

static ezp_gpu_event_footprint_t kernel_fp[2];
static unsigned pu_duration[2] = {0, 0};

bool much_greater_than (unsigned x, unsigned y)
{
  double MUCH_GREATER_FACTOR = 0.;
  char *env                  = getenv ("MUCH_GREATER_FACTOR");
  if (env != NULL)
    MUCH_GREATER_FACTOR = atof (env);
  return x > (y + MUCH_GREATER_FACTOR * y);
}

void life_init_ocl_hybrid (void)
{
  // OpenMP part (top half)
  kernel_fp[0].x = 0;
  kernel_fp[0].y = 0;
  kernel_fp[0].w = DIM;
  kernel_fp[0].h = (NB_TILES_Y / 2) * TILE_H; // Start with fifty-fifty;

  // GPU part (bottom half)
  kernel_fp[1].x = 0;
  kernel_fp[1].y = kernel_fp[0].y + kernel_fp[0].h;
  kernel_fp[1].w = DIM;
  kernel_fp[1].h = DIM - kernel_fp[1].y;

  life_init ();
}

// This function is called whenever the graphical window needs to be
// refreshed
void life_refresh_img (void)
{
  if (gpu_used) {
    const int gpu_size = DIM * kernel_fp[1].h;
    const int cpu_size = DIM * kernel_fp[0].h;
    cl_int err;

    // Read all the GPU side and write it into CPU buffer
    err = clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
                               cpu_size, gpu_size * sizeof (cell_t),
                               _table + cpu_size, 0, NULL, NULL);
    check (err, "Failed to read buffer from GPU");
  }

  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * LIFE_COLOR;
}

static inline void swap_tables (void)
{
  cell_t *tmp = _table;

  _table           = _alternate_table;
  _alternate_table = tmp;
}

///////////////////////////// Default tiling
int life_do_tile_default (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      if (j > 0 && j < DIM - 1 && i > 0 && i < DIM - 1) {

        unsigned n  = 0;
        unsigned me = cur_table (i, j);

        for (int yloc = i - 1; yloc < i + 2; yloc++)
          for (int xloc = j - 1; xloc < j + 2; xloc++)
            if (xloc != j || yloc != i)
              n += cur_table (yloc, xloc);

        if (me == 1 && n != 2 && n != 3) {
          me     = 0;
          change = 1;
        } else if (me == 0 && n == 3) {
          me     = 1;
          change = 1;
        }
        next_table (i, j) = me;
      }

  return change;
}

///////////////////////////// Default tiling

int life_do_tile_opt(int x, int y, int width, int height)
{
    if(x==0 || x == DIM-TILE_W)
      width -= 1;
    if(y==0 || y == DIM-TILE_H)
      height -= 1;
    x = x == 0 ? 1 : x;
    y = y == 0 ? 1 : y;

    int change = 0;
    int j = x;
    int next_j = x;
    cell_t *prev;
    cell_t *curr;
    cell_t *next;
    for (int i = y; i < y + height; i++) {
        for (; j < x + width - 6; j+=6) {
          prev = table_cell(_table, i-1, j-1);
          curr = table_cell(_table, i, j-1);
          next = table_cell(_table, i+1, j-1);

          unsigned n[8] = {
            0,
            prev[0] + prev[1] + prev[2] + curr[0] + curr[2] + next[0] + next[1] + next[2],
            prev[1] + prev[2] + prev[3] + curr[1] + curr[3] + next[1] + next[2] + next[3],
            prev[2] + prev[3] + prev[4] + curr[2] + curr[4] + next[2] + next[3] + next[4],
            prev[3] + prev[4] + prev[5] + curr[3] + curr[5] + next[3] + next[4] + next[5],
            prev[4] + prev[5] + prev[6] + curr[4] + curr[6] + next[4] + next[5] + next[6],
            prev[5] + prev[6] + prev[7] + curr[5] + curr[7] + next[5] + next[6] + next[7],
            0,
          };

          unsigned new_state[8] = {
            0,
            (n[1] == 3) | (curr[1] & (n[1] == 2)),
            (n[2] == 3) | (curr[2] & (n[2] == 2)),
            (n[3] == 3) | (curr[3] & (n[3] == 2)),
            (n[4] == 3) | (curr[4] & (n[4] == 2)),
            (n[5] == 3) | (curr[5] & (n[5] == 2)),
            (n[6] == 3) | (curr[6] & (n[6] == 2)),
            0,
          };
          change |= (curr[1] ^ new_state[1])
                  | (curr[2] ^ new_state[2])
                  | (curr[3] ^ new_state[3])
                  | (curr[4] ^ new_state[4])
                  | (curr[5] ^ new_state[5])
                  | (curr[6] ^ new_state[6]);

          next_table(i, j+0) = new_state[1];
          next_table(i, j+1) = new_state[2];
          next_table(i, j+2) = new_state[3];
          next_table(i, j+3) = new_state[4];
          next_table(i, j+4) = new_state[5];
          next_table(i, j+5) = new_state[6];
        }
        next_j = j;
        j = x;
      }

    for(int i = y; i < y + height; i++)
      for (int j = next_j; j < x + width; j++) {
          unsigned me = cur_table(i, j);
          unsigned n =  cur_table(i - 1, j - 1)
                      + cur_table(i - 1, j)
                      + cur_table(i - 1, j + 1)
                      + cur_table(i,     j - 1)
                      + cur_table(i,     j + 1)
                      + cur_table(i + 1, j - 1)
                      + cur_table(i + 1, j)
                      + cur_table(i + 1, j + 1);

          unsigned new_state = (n == 3) | (me & (n == 2));
          change |= (me ^ new_state);

          next_table(i, j) = new_state;
        }
    return change;
}


/////////////////////////////version Vector
#ifdef ENABLE_VECTO
#include <immintrin.h>

#if __AVX2__== 1
// #warning "AVX2 is enabled"



// int life_do_tile_avx (int x, int y, int width, int height)
// {
//   int change = 0;

//   for (int i = y; i < y + height; i++) {
//     for (int j = x; j <= x + width - AVX_VEC_SIZE_FLOAT;
//          j += AVX_VEC_SIZE_FLOAT) {
//       if (j <= 0 || j + AVX_VEC_SIZE_FLOAT > DIM || i <= 0 || i >= DIM - 1) {
//         // Scalar fallback for the 8 cells
//         for (int k = 0; k < AVX_VEC_SIZE_FLOAT; k++) {
//           int jj = j + k;
//           if (jj > 0 && jj < DIM - 1 && i > 0 && i < DIM - 1) {
//             unsigned me = cur_table (i, jj);
//             unsigned n  = cur_table (i - 1, jj - 1) + cur_table (i - 1, jj) +
//                          cur_table (i - 1, jj + 1) + cur_table (i, jj - 1) +
//                          cur_table (i, jj + 1) + cur_table (i + 1, jj - 1) +
//                          cur_table (i + 1, jj) + cur_table (i + 1, jj + 1);

//             unsigned new_state = (n == 3) || (me && n == 2);
//             change |= (me != new_state);
//             next_table (i, jj) = new_state;
//           }
//         }
//       } else {
//         // Vectorized core
//         __m256i top_left = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + (i - 1) * DIM + (j - 1))));
//         __m256i top_center = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + (i - 1) * DIM + j)));
//         __m256i top_right = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + (i - 1) * DIM + (j + 1))));

//         __m256i mid_left = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + i * DIM + (j - 1))));
//         __m256i mid_center = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + i * DIM + j)));
//         __m256i mid_right = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + i * DIM + (j + 1))));

//         __m256i bot_left = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + (i + 1) * DIM + (j - 1))));
//         __m256i bot_center = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + (i + 1) * DIM + j)));
//         __m256i bot_right = _mm256_cvtepu8_epi32 (
//             _mm_loadl_epi64 ((__m128i *)(_table + (i + 1) * DIM + (j + 1))));

//         __m256i sum = top_left + top_center + top_right + mid_left + mid_right +
//                       bot_left + bot_center + bot_right;

//         __m256i one  = _mm256_set1_epi32 (1);
//         __m256i prev = mid_center & one;

//         __m256i cmp3      = _mm256_cmpeq_epi32 (sum, _mm256_set1_epi32 (3));
//         __m256i cmp2      = _mm256_cmpeq_epi32 (sum, _mm256_set1_epi32 (2));
//         __m256i cond      = prev & cmp2;
//         __m256i new_state = cmp3 | cond;

//         new_state = new_state & one;

//         __m256i diff = new_state ^ prev;

//         change |= !_mm256_testz_si256 (diff, diff);

//         __attribute__ ((aligned (32))) uint32_t new_vals[8];
//         _mm256_store_si256 ((__m256i *)new_vals, new_state);
//         for (int k = 0; k < 8; k++) {
//           _alternate_table[i * DIM + j + k] = (uint8_t)new_vals[k];
//         }
//       }
//     }
//   }

//   return change;
// }

int life_do_tile_AVX(int x, int y, int width, int height)
{
  int change = 0;

  // Constantes AVX2 utilisées dans la boucle
  const __m256i const_1 = _mm256_set1_epi8(1);
  const __m256i const_2 = _mm256_set1_epi8(2);
  const __m256i const_3 = _mm256_set1_epi8(3);

  for (int i = y; i < y + height; i++) {
    for (int j = x; j < x + width; j += 32) {

      if (j > 0 && j + 31 < DIM - 1 && i > 0 && i < DIM - 1) {
        // Chargement des 8 voisins
        __m256i above_left   = _mm256_loadu_si256((__m256i*)&_table[(i - 1) * DIM + (j - 1)]);
        __m256i above_mid    = _mm256_loadu_si256((__m256i*)&_table[(i - 1) * DIM + (j)]);
        __m256i above_right  = _mm256_loadu_si256((__m256i*)&_table[(i - 1) * DIM + (j + 1)]);

        __m256i middle_left  = _mm256_loadu_si256((__m256i*)&_table[i * DIM + (j - 1)]);
        __m256i middle       = _mm256_loadu_si256((__m256i*)&_table[i * DIM + j]);
        __m256i middle_right = _mm256_loadu_si256((__m256i*)&_table[i * DIM + (j + 1)]);

        __m256i below_left   = _mm256_loadu_si256((__m256i*)&_table[(i + 1) * DIM + (j - 1)]);
        __m256i below_mid    = _mm256_loadu_si256((__m256i*)&_table[(i + 1) * DIM + (j)]);
        __m256i below_right  = _mm256_loadu_si256((__m256i*)&_table[(i + 1) * DIM + (j + 1)]);

        // Somme des 8 voisins
        __m256i sum1 = _mm256_adds_epu8(above_left, above_mid);
        sum1 = _mm256_adds_epu8(sum1, above_right);
        __m256i sum2 = _mm256_adds_epu8(middle_left, middle_right);
        __m256i sum3 = _mm256_adds_epu8(below_left, below_mid);
        sum3 = _mm256_adds_epu8(sum3, below_right);
        __m256i sum = _mm256_adds_epu8(sum1, sum2);
        sum = _mm256_adds_epu8(sum, sum3);

        // Application des règles
        __m256i is_two   = _mm256_cmpeq_epi8(sum, const_2);
        __m256i is_three = _mm256_cmpeq_epi8(sum, const_3);
        __m256i alive    = _mm256_cmpeq_epi8(middle, const_1);

        // Normalize to 0/1 (important!)
        is_two = _mm256_and_si256(is_two, const_1);
        is_three = _mm256_and_si256(is_three, const_1);
        alive = _mm256_and_si256(alive, const_1);

        // Calcul des nouveaux états
        __m256i stay_alive = _mm256_or_si256(is_two, is_three);
        __m256i alive_next = _mm256_and_si256(alive, stay_alive);
        __m256i born = _mm256_andnot_si256(alive, is_three);
        __m256i final = _mm256_or_si256(alive_next, born);

        // Détection de changement
        __m256i changed = _mm256_xor_si256(middle, final);
        if (!_mm256_testz_si256(changed, changed)) {
          change = 1;
        }

        // Stockage
        _mm256_storeu_si256((__m256i*)&_alternate_table[i * DIM + j], final);

      } else {
        // Fallback scalaire pour les bords
        for (int k = 0; k < 32 && (j + k) < (x + width); k++) {
          int jj = j + k;

          if (jj > 0 && jj < DIM - 1 && i > 0 && i < DIM - 1) {
            uint8_t me = _table[i * DIM + jj];
            uint8_t n =
              _table[(i - 1) * DIM + (jj - 1)] + _table[(i - 1) * DIM + jj] + _table[(i - 1) * DIM + (jj + 1)] +
              _table[i * DIM + (jj - 1)]                                   + _table[i * DIM + (jj + 1)] +
              _table[(i + 1) * DIM + (jj - 1)] + _table[(i + 1) * DIM + jj] + _table[(i + 1) * DIM + (jj + 1)];

            uint8_t new_state = (n == 3) || (me && n == 2);
            if (me != new_state) change = 1;
            _alternate_table[i * DIM + jj] = new_state;
          }
        }
      }
    }
  }

  return change;
}

#endif // AVX
#endif

///////////////////////////////////////////////////////////////////////////


///////////////////////////// Sequential version (seq)
//
unsigned life_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    int change = life_do_tile_default (0, 0, DIM, DIM);

    if (!change)
      return it;

    swap_tables ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
//
unsigned life_compute_tiled (unsigned nb_iter)
{
  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

    for (int y = 0; y < DIM; y += TILE_H) {
      change |= life_do_tile_default (0, y, TILE_W, TILE_H);
      change |= life_do_tile_default (DIM - TILE_W, y, TILE_W, TILE_H);
    }
    for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W) {
      change |= life_do_tile_default (x, 0, TILE_W, TILE_H);
      change |= life_do_tile_default (x, DIM - TILE_H, TILE_W, TILE_H);
    }

    for (int y = TILE_H; y < DIM - TILE_H; y += TILE_H)
      for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W)
        change |= do_tile (x, y, TILE_W, TILE_H);

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

unsigned life_compute_lazy(unsigned nb_iter) {
    unsigned res = 0;
    bool *tmp;
    for (unsigned it = 1; it <= nb_iter; it++) {
      unsigned change = 0;

      for (int j = 0; j < TILES_Y; j++) {
        for (int i = 0; i < TILES_X; i++) {
          if (!table_tile(active, j, i))
            continue;
          if (do_tile(i * TILE_W, j * TILE_H, TILE_W, TILE_H)) {
            change = 1;
            for (int dj = -1; dj <= 1; dj++) {
              for (int di = -1; di <= 1; di++) {
                int nj = j + dj;
                int ni = i + di;
                if (nj >= 0 && nj < TILES_Y && ni >= 0 && ni < TILES_X) {
                    table_tile(new_active, nj, ni) = true;
                }
              }
            }
          }
        }
      }

      swap_tables();
      if (!change) {
          res = it;
          break;
      }

      // Préparation pour la prochaine itération : on passe les nouvelles activations

      tmp = active;
      active = new_active;
      new_active = tmp;
      bzero(new_active, TILES_X * TILES_Y * sizeof(bool));
    }

    return res;
}

///////////////////////////// Tiled ompfor version (ompfor)
//
unsigned life_compute_ompfor (unsigned nb_iter)
{
  unsigned res = 0;
  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

#pragma omp parallel shared(change)
    {
#pragma omp for schedule(runtime) nowait
      for (int y = 0; y < DIM; y += TILE_H) {
        change |= life_do_tile_default (0, y, TILE_W, TILE_H);
        change |= life_do_tile_default (DIM - TILE_W, y, TILE_W, TILE_H);
      }
#pragma omp for schedule(runtime) nowait
      for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W) {
        change |= life_do_tile_default (x, 0, TILE_W, TILE_H);
        change |= life_do_tile_default (x, DIM - TILE_H, TILE_W, TILE_H);
      }

#pragma omp for schedule(runtime)
      for (int y = TILE_H; y < DIM - TILE_H; y += TILE_H)
        for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W)
          change |= do_tile (x, y, TILE_W, TILE_H);
    }

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

///////////////////////////// Tiled ompfor version (ompfor)
//
unsigned life_compute_ompcol2 (unsigned nb_iter)
{
  unsigned res = 0;
  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

#pragma omp parallel shared(change)
    {
#pragma omp for schedule(runtime) nowait
      for (int y = 0; y < DIM; y += TILE_H) {
        change |= life_do_tile_default (0, y, TILE_W, TILE_H);
        change |= life_do_tile_default (DIM - TILE_W, y, TILE_W, TILE_H);
      }
#pragma omp for schedule(runtime) nowait
      for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W) {
        change |= life_do_tile_default (x, 0, TILE_W, TILE_H);
        change |= life_do_tile_default (x, DIM - TILE_H, TILE_W, TILE_H);
      }

#pragma omp for collapse(2) schedule(runtime)
      for (int y = TILE_H; y < DIM - TILE_H; y += TILE_H)
        for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W)
          change |= do_tile (x, y, TILE_W, TILE_H);
    }

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

///////////////////////////// Tiled lazy ompfor version (omplazy)

unsigned life_compute_omplazy(unsigned nb_iter) {
    unsigned res = 0;
    bool *tmp;
    for (unsigned it = 1; it <= nb_iter; it++) {
      unsigned change = 0;

#pragma omp parallel for shared(active, new_active, change) schedule(runtime)  \
    collapse(2)
      for (int j = 0; j < TILES_Y; j++) {
        for (int i = 0; i < TILES_X; i++) {
          if (!table_tile(active, j, i))
            continue;
          if (do_tile(i * TILE_W, j * TILE_H, TILE_W, TILE_H)) {
            change = 1;
            for (int dj = -1; dj <= 1; dj++) {
              for (int di = -1; di <= 1; di++) {
                int nj = j + dj;
                int ni = i + di;
                if (nj >= 0 && nj < TILES_Y && ni >= 0 && ni < TILES_X) {
                    table_tile(new_active, nj, ni) = true;
                }
              }
            }
          }
        }
      }

      swap_tables();
      if (!change) {
          res = it;
          break;
      }

      // Préparation pour la prochaine itération : on passe les nouvelles activations

      tmp = active;
      active = new_active;
      new_active = tmp;
      bzero(new_active, TILES_X * TILES_Y * sizeof(bool));
    }

    return res;
}

///////////////////////////// Tiled taskloop version (taskloop)
//
unsigned life_compute_taskloop (unsigned nb_iter)
{
  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

#pragma omp parallel shared(change) // Creates threads
    {
#pragma omp single nowait // Only one of them will generate
#pragma omp taskloop      // the tasks created by taskloop
      for (int y = 0; y < DIM; y += TILE_H) {
        change |= life_do_tile_default (0, y, TILE_W, TILE_H);
        change |= life_do_tile_default (DIM - TILE_W, y, TILE_W, TILE_H);
      }
#pragma omp single nowait // Only one of them will generate
#pragma omp taskloop      // the tasks created by taskloop
      for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W) {
        change |= life_do_tile_default (x, 0, TILE_W, TILE_H);
        change |= life_do_tile_default (x, DIM - TILE_H, TILE_W, TILE_H);
      }

#pragma omp single nowait // Only one of them will generate
#pragma omp taskloop      // the tasks created by taskloop
      for (int y = TILE_H; y < DIM - TILE_H; y += TILE_H)
        for (int x = TILE_W; x < DIM - TILE_W; x += TILE_W)
          change |= do_tile (x, y, TILE_W, TILE_H);
    }

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

///////////////////////////// Version OCL hybrid

unsigned ocl_hybrid_omp ()
{
  unsigned change = 0;

#pragma omp parallel for collapse(2) schedule(runtime)
  for (int y = kernel_fp[0].y; y < kernel_fp[0].y + 1 + kernel_fp[0].h;
       y += TILE_H)
    for (int x = kernel_fp[0].x; x < kernel_fp[0].x + kernel_fp[0].w;
         x += TILE_W)
      change |= do_tile (x, y, TILE_W, TILE_H);

  return change;
}

void ocl_hybrid_ocl_launch (size_t global[2], size_t local[2])
{
  cl_int err = 0;

  // Set kernel arguments
  err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                         &ocl_cur_buffer (0));
  err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                         &ocl_next_buffer (0));
  err |= clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (unsigned),
                         &kernel_fp[1].y);
  check (err, "Failed to set kernel arguments");

  // Launch GPU kernel
  err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2, NULL,
                                global, local, 0, NULL,
                                ezp_ocl_eventptr (EVENT_START_KERNEL, 0));
  check (err, "Failed to execute kernel");
  clFlush (ocl_queue (0));
}

void ocl_hybrid_exchange_borders ()
{
  size_t cpu_size = DIM * kernel_fp[0].h;
  cl_int err      = 0;

  // Write GPU side border to CPU buffer
  err =
      clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, cpu_size,
                           LINE_SIZE, _table + cpu_size, 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");

  // Write CPU side border to GPU buffer
  err = clEnqueueWriteBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
                              cpu_size - DIM, LINE_SIZE,
                              _table + cpu_size - DIM, 0, NULL, NULL);
  check (err, "Failed to write buffer to GPU");
}

void ocl_hybrid_load_balancing (size_t global[2])
{
  size_t cpu_size = DIM * kernel_fp[0].h;
  cl_int err      = 0;

  if (pu_duration[1] != 0) {
    if (much_greater_than (pu_duration[1], pu_duration[0]) &&
        kernel_fp[1].h > TILE_H) {

      // Write GPU transferred zone to CPU buffer
      err = clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
                                 cpu_size, TILE_H * LINE_SIZE,
                                 _table + cpu_size, 0, NULL, NULL);
      check (err, "Failed to read buffer from GPU");

      // Give more to CPU
      kernel_fp[0].h += TILE_H;
      kernel_fp[1].h -= TILE_H;
      kernel_fp[1].y += TILE_H;
      global[1] = kernel_fp[1].h;

    } else if (much_greater_than (pu_duration[0], pu_duration[1]) &&
               kernel_fp[0].h > TILE_H) {

      // Write GPU transferred zone to CPU buffer
      err = clEnqueueWriteBuffer (
          ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, cpu_size - DIM * TILE_H,
          TILE_H * LINE_SIZE, _table + cpu_size - DIM * TILE_H, 0, NULL, NULL);
      check (err, "Failed to write buffer from GPU");

      // Give more to GPU
      kernel_fp[0].h -= TILE_H;
      kernel_fp[1].h += TILE_H;
      kernel_fp[1].y -= TILE_H;
      global[1] = kernel_fp[1].h;
    }
  }
}

unsigned life_compute_ocl_hybrid (unsigned nb_iter)
{
  char change      = 0;
  unsigned res     = 0;
  size_t global[2] = {DIM,
                      kernel_fp[1].h}; // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  uint64_t clock;

  for (unsigned it = 1; it <= nb_iter; it++) {

    ocl_hybrid_load_balancing (global);
    ocl_hybrid_exchange_borders ();
    ocl_hybrid_ocl_launch (global, local);

    clock = ezp_gettime (); // Take reference time

    change = ocl_hybrid_omp (); // Compute CPU part

    { // Retrieve execution times
      pu_duration[0] = ezp_gettime () - clock;

      clFinish (ocl_queue (0)); // Wait for GPU to finish
      pu_duration[1] = ezp_gpu_event_monitor (
          0, EVENT_START_KERNEL, clock, &kernel_fp[1], TASK_TYPE_COMPUTE, 0);
      ezp_gpu_event_reset ();
    }

    { // Swap buffers
      swap_tables ();
      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
    }

    // if (!change) { // we stop if all cells are stable
    //   res = it;
    //   break;
    // }
  }
  return res;
}
///////////////////////////// Initial configs

void life_draw_guns (void);

static inline void set_cell (int y, int x)
{
  cur_table (y, x) = 1;
  if (gpu_used)
    *((char *)image + y * DIM + x) = 1;
}

static inline int get_cell (int y, int x)
{
  return cur_table (y, x);
}

static void inline life_rle_parse (char *filename, int x, int y,
                                   int orientation)
{
  rle_lexer_parse (filename, x, y, set_cell, orientation);
}

static void inline life_rle_generate (char *filename, int x, int y, int width,
                                      int height)
{
  rle_generate (x, y, width, height, get_cell, filename);
}

void life_draw (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_rle_parse (param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper (param, life_draw_guns);
}

static void otca_autoswitch (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void otca_life (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners (char *filename, int distance)
{
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_HINVERT);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_VINVERT);
  life_rle_parse (filename, distance, distance,
                  RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_off -ts 64 -r 10 -si
void life_draw_otca_off (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_on -ts 64 -r 10 -si
void life_draw_otca_on (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_draw_meta3x3 (void)
{
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life (j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                 1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life -a bugs -ts 64
void life_draw_bugs (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life -v omp -a ship -s 512 -m -ts 16
void life_draw_ship (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }

  for (int y = 43; y < DIM - 134; y += 148) {
    life_rle_parse ("data/rle/greyship.rle", DIM - 100, y,
                    RLE_ORIENTATION_NORMAL);
  }
}

void life_draw_stable (void)
{
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell (i, j);
      set_cell (i, j + 1);
      set_cell (i + 1, j);
      set_cell (i + 1, j + 1);
    }
}

void life_draw_oscil (void)
{
  for (int i = 2; i < DIM - 4; i += 4)
    for (int j = 2; j < DIM - 4; j += 4) {
      if ((j - 2) % 8) {
        set_cell (i + 1, j);
        set_cell (i + 1, j + 1);
        set_cell (i + 1, j + 2);
      } else {
        set_cell (i, j + 1);
        set_cell (i + 1, j + 1);
        set_cell (i + 2, j + 1);
      }
    }
}

void life_draw_guns (void)
{
  at_the_four_corners ("data/rle/gun.rle", 1);
}

static unsigned long seed = 123456789;

// Deterministic function to generate pseudo-random configurations
// independently of the call context
static unsigned long pseudo_random ()
{
  unsigned long a = 1664525;
  unsigned long c = 1013904223;
  unsigned long m = 4294967296;

  seed = (a * seed + c) % m;
  seed ^= (seed >> 21);
  seed ^= (seed << 35);
  seed ^= (seed >> 4);
  seed *= 2685821657736338717ULL;
  return seed;
}

void life_draw_random (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (pseudo_random () & 1)
        set_cell (i, j);
}

// Suggested cmdline: ./run -k life -a clown -s 256 -i 110
void life_draw_clown (void)
{
  life_rle_parse ("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}

void life_draw_diehard (void)
{
  life_rle_parse ("data/rle/diehard.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}

static void dump (int size, int x, int y)
{
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      if (get_cell (i, j))
        set_cell (i + x, j + y);
}

static void moult_rle (int size, int p, char *filepath)
{

  int positions = (DIM) / (size + 1);

  life_rle_parse (filepath, size / 2, size / 2, RLE_ORIENTATION_NORMAL);
  for (int k = 0; k < p; k++) {
    int px = pseudo_random () % positions;
    int py = pseudo_random () % positions;
    dump (size, px * size, py * size);
  }
}

// ./run  -k life -a moultdiehard130  -v omp -ts 32 -m -s 512
void life_draw_moultdiehard130 (void)
{
  moult_rle (16, 128, "data/rle/diehard.rle");
}

// ./run  -k life -a moultdiehard2474  -v omp -ts 32 -m -s 1024
void life_draw_moultdiehard1398 (void)
{
  moult_rle (52, 96, "data/rle/diehard1398.rle");
}

// ./run  -k life -a moultdiehard2474  -v omp -ts 32 -m -s 2048
void life_draw_moultdiehard2474 (void)
{
  moult_rle (104, 32, "data/rle/diehard2474.rle");
}

// Just in case we want to draw an initial configuration and dump it to file,
// with no iteration at all
unsigned life_compute_none (unsigned nb_iter)
{
  return 1;
}

//////////// debug ////////////
static int debug_hud = -1;

void life_config (char *param)
{
  seed += param ? atoi (param) : 0;
  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], debug_hud);
  }
}

void life_debug (int x, int y)
{
  if (x == -1 || y == -1)
    ezv_hud_set (ctx[0], debug_hud, NULL);
  else {
    ezv_hud_set (ctx[0], debug_hud, cur_table (y, x) ? "Alive" : "Dead");
  }
}

//Only called when --dump or --thumbnails is used
void life_refresh_img_ocl (void)
{
  const int size = DIM * LINE_SIZE;
  cl_int err;

  err = clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0,
                             size, _table, 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");

  life_refresh_img ();
}



#ifdef ENABLE_MPI
#include <mpi.h>

static int rank, size;


void life_init_mpi() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    life_init(); // Tous les processus initialisent la grille complète
}

static int rankSize(int rank) { 
    return DIM / size + (rank < DIM % size); 
}

static int rankTop(int rank) { 
    return rank * (DIM / size) + (rank < DIM % size ? rank : DIM % size); 
}

void life_refresh_img_mpi()
{
    MPI_Status status;

    if (rank == 0) {
        // Affiche sa propre partie
        #pragma omp parallel
        {
        #pragma omp taskloop
        for (int y = 0; y < rankSize(0); y++)
            for (int x = 0; x < DIM; x++)
                cur_img(y, x) = cur_table(y, x) * LIFE_COLOR;
        }
        // Réception des autres parties
        for (int i = 1; i < size; i++) {
            MPI_Recv(&(_table[rankTop(i) * DIM]), rankSize(i) * DIM, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
        }

        // Affichage de l'image complète
        #pragma omp parallel
        {
        #pragma omp taskloop
        for (int y = 0; y < DIM; y++)
            for (int x = 0; x < DIM; x++)
                cur_img(y, x) = cur_table(y, x) * LIFE_COLOR;
        }
    } else {
        // Chaque processus envoie uniquement sa partie calculée à 0
        MPI_Send(&(_table[rankTop(rank) * DIM]), rankSize(rank) * DIM, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    //Affichage debug
    // printf("Rank %d: lines %d to %d\n", rank, rankTop(rank), rankTop(rank) + rankSize(rank) - 1);
}

// --- Échange des bords entre processus voisins ---
void exchange_borders()
{
    MPI_Status status;
    // Envoi et réception des lignes frontières
    // Tableaux temporaires pour les lignes à envoyer/recevoir
    char *send_top = &cur_table(rankTop(rank), 0);
    char *recv_top = &cur_table(rankTop(rank) - 1, 0);
    char *send_bottom = &cur_table(rankTop(rank) + rankSize(rank) - 1, 0);
    char *recv_bottom = &cur_table(rankTop(rank) + rankSize(rank), 0);

    // Échange avec le rang supérieur (rank-1)
    if (rank > 0) {
        MPI_Sendrecv(send_top, DIM, MPI_CHAR, rank - 1, 0,
                     recv_top, DIM, MPI_CHAR, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
    }
    // Échange avec le rang inférieur (rank+1)
    if (rank < size - 1) {
        MPI_Sendrecv(send_bottom, DIM, MPI_CHAR, rank + 1, 1,
                     recv_bottom, DIM, MPI_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

unsigned life_compute_mpi(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // Échange des bords avec les voisins
        exchange_borders();
        int change = 0;
        #pragma omp parallel for schedule(runtime)
        for(int y = rankTop(rank); y < rankTop(rank) + rankSize(rank); y += TILE_H){
          for(int x = 0; x < DIM; x += TILE_W){
            change  |= do_tile(x, y, TILE_W, TILE_H);
          }
        }

        // Synchronisation du critère d'arrêt
        int global_change;
        MPI_Allreduce(&change, &global_change, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        if (!global_change)
            return it;
        swap_tables();
    }
    return 0;
}



void life_init_mpi_col() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  life_init(); // Tous les processus initialisent la grille complète
}

unsigned life_compute_mpi_col(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
      int change = 0;
      exchange_borders();
      #pragma omp parallel for collapse(2) schedule(runtime)
      for(int y = rankTop(rank); y < rankTop(rank) + rankSize(rank); y += TILE_H){
        for(int x = 0; x < DIM; x += TILE_W){
          change  |= do_tile(x, y, TILE_W, TILE_H);
        }
      }
        
      // Synchronisation globale du critère d'arrêt
      int global_change;
      MPI_Allreduce(&change, &global_change, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
      if (!global_change) 
        return it;
      
      swap_tables();
    }
    return 0;
}

void life_refresh_img_mpi_col() {
    int counts[size], displs[size];
    for (int i = 0; i < size; i++) {
        counts[i] = rankSize(i) * DIM;
        displs[i] = rankTop(i) * DIM;
    }

    // Seul le processus 0 met à jour l'image complète
    if (rank == 0) {
      #pragma omp parallel
      {
      #pragma omp taskloop
      for (int y = 0; y < DIM; y++) {
          for (int x = 0; x < DIM; x++) {
              cur_img(y, x) = cur_table(y, x) * LIFE_COLOR;
          }
      }
      }
    } else {
      #pragma omp parallel
      {
      #pragma omp taskloop
      for (int y = 0; y < rankSize(rank); y++)
        for (int x = 0; x < DIM; x++)
          cur_img(rankTop(rank) + y, x) = cur_table(rankTop(rank) + y, x) * LIFE_COLOR;
      }
    }


    // Détermine l'adresse source pour l'envoi
    char* sendbuf = (rank == 0) 
    ? MPI_IN_PLACE  // Le root ne copie rien (les données sont déjà en place)
    : &_table[rankTop(rank) * DIM];  // Les autres envoient leur portion

    // Appel à MPI_Gather pour rassembler toutes les portions du tableau sur le rang 0
    MPI_Gatherv(
    sendbuf,        // Données à envoyer
    counts[rank],   // Taille des données à envoyer
    MPI_CHAR,       // Type des données envoyées
    _table,         // Destination sur le root
    counts,         // Tailles de chaque bloc reçu
    displs,         // Décalages dans le buffer de réception
    MPI_CHAR,       // Type des données reçues
    0,              // Rang du processus root
    MPI_COMM_WORLD  // Communicateur
    );

}
#endif
