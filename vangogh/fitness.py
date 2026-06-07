import numpy as np
from PIL import Image
from imgcompare import image_diff
from multiprocess import Pool, cpu_count
from scipy.spatial import KDTree
from vangogh.util import NUM_VARIABLES_PER_POINT

QUERY_POINTS = []


def draw_voronoi_matrix(genotype, img_width, img_height, scale=1):
    scaled_img_width = int(img_width * scale)
    scaled_img_height = int(img_height * scale)
    num_points = int(len(genotype) / NUM_VARIABLES_PER_POINT)

    coords = []
    colors = []
    for r in range(num_points):
        p = r * NUM_VARIABLES_PER_POINT
        x, y, red, green, blue = genotype[p:p + NUM_VARIABLES_PER_POINT]
        coords.append((x * scale, y * scale))
        colors.append((red, green, blue))

    voronoi_kdtree = KDTree(coords)

    # Vectorized query — no Python pixel loop
    xs = np.arange(scaled_img_width)
    ys = np.arange(scaled_img_height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    query_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    _, region_indices = voronoi_kdtree.query(query_points)

    colors_array = np.array(colors, dtype='uint8')
    data = colors_array[region_indices].reshape(scaled_img_height, scaled_img_width, 3)
    return data


def draw_voronoi_image(genotype, img_width, img_height, scale=1) -> Image:
    data = draw_voronoi_matrix(genotype, img_width, img_height, scale)
    return Image.fromarray(data, 'RGB')


def compute_difference(genotype, reference_image: Image):
    actual = draw_voronoi_matrix(genotype, reference_image.width, reference_image.height)
    return image_diff(Image.fromarray(actual, 'RGB'), reference_image)


def compute_grid_fitness(genotype, reference_image: Image, grid_cell_size):
    generated = draw_voronoi_matrix(genotype, reference_image.width, reference_image.height)
    reference_array = np.array(reference_image, dtype=np.float32)
    pixel_error = np.mean(np.abs(reference_array - generated.astype(np.float32)), axis=2)

    num_cells_y = (reference_image.height + grid_cell_size - 1) // grid_cell_size
    num_cells_x = (reference_image.width + grid_cell_size - 1) // grid_cell_size
    grid_fitness = np.zeros((num_cells_y, num_cells_x))

    for cy in range(num_cells_y):
        for cx in range(num_cells_x):
            cell = pixel_error[
                cy * grid_cell_size:min((cy + 1) * grid_cell_size, reference_image.height),
                cx * grid_cell_size:min((cx + 1) * grid_cell_size, reference_image.width)
            ]
            grid_fitness[cy, cx] = np.mean(cell) if cell.size > 0 else 0.0

    return grid_fitness, (num_cells_y, num_cells_x)


def worker(args):
    return compute_difference(args[0], args[1])


def drawing_fitness_function(genes, reference_image: Image):
    with Pool(min(max(cpu_count() - 1, 1), 4)) as p:
        fitness_values = list(p.map(worker, zip(genes, [reference_image] * genes.shape[0])))
    return np.array(fitness_values)