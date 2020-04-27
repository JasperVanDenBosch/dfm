import numpy


def reframe(a, width=100, height=100, x=0, y=0):
    """Insert all or part of a 2D array inside an empty 2D array,
    centered on a given location.

    By default puts a at the top-left corner of the output, which will be 100 by 100.

    Args:
        a (numpy.ndarray): array to be inserted
        width (int, optional): width of the output. Defaults to 100.
        height (int, optional): height of the output. Defaults to 100.
        x (int, optional): First index of the output at which to center a. Defaults to 0.
        y (int, optional): Second index of the output at which to center a. Defaults to 0.

    Returns:
        numpy.ndarray: Array (2D) of zeros except where the input is placed
    """
    assert len(a.shape) == 2    ## must be two dimensional
    assert a.shape[0] % 2 != 0  ## width must be odd
    assert a.shape[1] % 2 != 0  ## height must be odd
    ## coordinates must be in output range:
    assert x >= 0
    assert y >= 0
    assert x < width
    assert y < height
    assert width > 0
    assert height > 0
    y_extent, x_extent = [int((dim_size - 1) / 2) for dim_size in a.shape]
    canvas = numpy.zeros([height + 2 * y_extent, width + 2 * x_extent])
    canvas[y: y + a.shape[1], x: x + a.shape[0]] = a
    return canvas[y_extent: y_extent + height, x_extent: x_extent + width]
