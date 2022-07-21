class Basis():
  """Abstract class of orbital functions."""

  def __init__(self):
    """Abstract initializer of basis."""

  def __call__(self, x):
    """Compute the basis functions.

    Args:
      x: shape is [..., 3]
    Returns:
      output: shape equal to [..., num_basis],
        where the batch dims are equal to x
    """
    raise NotImplementedError('__call__ function has not been implemented')

  def overlap(self):
    """Compute the overlap between basis functions.

    Returns:
      output: shape equal to [num_basis, num_basis]
    """
    raise NotImplementedError('overlap function has not been implemented.')

  def init(self):
    """Initialize the parameter of this class if any."""
