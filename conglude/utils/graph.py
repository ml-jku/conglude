import torch
from scipy.special import gammainc
from torch.distributions.normal import Normal



def random_rotation_matrix():
    """
    Generate a random 3×3 rotation matrix.

    Returns
    -------
    torch.Tensor
        A (3, 3) orthonormal rotation matrix.
    """

    theta = 2 * torch.pi * torch.rand(1)  # Random rotation around the z-axis
    phi = torch.acos(2 * torch.rand(1) - 1)  # Random rotation around the y-axis
    psi = 2 * torch.pi * torch.rand(1)  # Random rotation around the x-axis

    Rz = torch.Tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    Ry = torch.Tensor(
        [[torch.cos(phi), 0, torch.sin(phi)], [0, 1, 0], [-torch.sin(phi), 0, torch.cos(phi)]]
    )
    Rx = torch.Tensor(
        [[1, 0, 0], [0, torch.cos(psi), -torch.sin(psi)], [0, torch.sin(psi), torch.cos(psi)]]
    )
    R = torch.mm(Rz, torch.mm(Ry, Rx))  # Combined rotation matrix

    return R



def sample_fibonacci_grid(
    centroid: torch.Tensor,
    radius: torch.Tensor,
    num_points: int,
    random_rotations: bool = True,
) -> torch.Tensor:
    """
    Sample approximately uniformly distributed points on a sphere surface
    using a Fibonacci spiral.

    Parameters
    ----------
    centroid: torch.Tensor
        Tensor of shape (3,) representing the center of the sphere.
    radius: torch.Tensor or float
        Radius of the sphere.
    num_points: int
        Number of points to sample on the sphere surface.
    random_rotations: bool, optional
        If True, apply a random 3D rotation to the sampled grid to avoid fixed orientation bias.

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_points, 3) containing sampled 3D coordinates.
    """

    golden_ratio = (1.0 + torch.sqrt(torch.tensor(5.0))) / 2.0

    theta = 2 * torch.pi * torch.arange(num_points).float() / golden_ratio
    phi = torch.acos(1 - 2 * (torch.arange(num_points).float() + 0.5) / num_points)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1)
    if random_rotations:
        rotation_matrix = random_rotation_matrix()
        points = torch.mm(points, rotation_matrix.T)  # Corrected rotation step

    points = centroid + points

    return points



def sample_uniform_in_sphere(centroid: torch.Tensor, radius: torch.Tensor, num_points: int):
    """
    Sample points uniformly inside a 3D sphere.

    Parameters
    ----------
    centroid: torch.Tensor
        Tensor of shape (3,) representing the center of the sphere.
    radius: torch.Tensor or float
        Radius of the sphere.
    num_points: int
        Number of points to sample inside the sphere.

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_points, 3) containing sampled 3D coordinates uniformly distributed within the sphere.
    """

    r = radius
    ndim = centroid.size(0)
    normal_dist = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    x = normal_dist.sample((num_points, ndim)).squeeze(-1)
    ssq = torch.sum(x**2, axis=1)
    fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / torch.sqrt(ssq)
    frtiled = fr.unsqueeze(1).repeat(1, ndim)
    p = centroid + x * frtiled

    return p.clone()