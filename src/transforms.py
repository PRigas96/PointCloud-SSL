import torch
import torch.nn.functional as F
# ALL transformation must have an input vector and work for batches

class Rotation(object):
    def __init__(self, angles):
        # angles: (batch_size, 3)
        self.yaw = angles[:, 0]
        self.pitch = angles[:, 1]
        self.roll = angles[:, 2]

    def get_rotation_matrix(self, batch_size):
        yaw_matrix = torch.stack([
            torch.stack([torch.cos(self.yaw), torch.zeros(batch_size), torch.sin(self.yaw)], dim=1),
            torch.stack([torch.zeros(batch_size), torch.ones(batch_size), torch.zeros(batch_size)], dim=1),
            torch.stack([-torch.sin(self.yaw), torch.zeros(batch_size), torch.cos(self.yaw)], dim=1)
        ], dim=1)

        pitch_matrix = torch.stack([
            torch.stack([torch.ones(batch_size), torch.zeros(batch_size), torch.zeros(batch_size)], dim=1),
            torch.stack([torch.zeros(batch_size), torch.cos(self.pitch), -torch.sin(self.pitch)], dim=1),
            torch.stack([torch.zeros(batch_size), torch.sin(self.pitch), torch.cos(self.pitch)], dim=1)
        ], dim=1)

        roll_matrix = torch.stack([
            torch.stack([torch.cos(self.roll), -torch.sin(self.roll), torch.zeros(batch_size)], dim=1),
            torch.stack([torch.sin(self.roll), torch.cos(self.roll), torch.zeros(batch_size)], dim=1),
            torch.stack([torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size)], dim=1)
        ], dim=1)

        rotation_matrix = torch.bmm(torch.bmm(yaw_matrix, pitch_matrix), roll_matrix)
        return rotation_matrix

    def __call__(self, point_cloud):
        batch_size = point_cloud.size(0)
        rotation_matrix = self.get_rotation_matrix(batch_size)
        return torch.bmm(point_cloud, rotation_matrix) 

class Reflection(object):
    def __init__(self, axis):
        # axis: (batch_size, 3)
        self.axis = axis # axis is a vector of length 3, with vals -1 or 1

    def __call__(self, point_cloud):
        return point_cloud * self.axis

class Shear(object):
    def __init__(self, shear_factors):
        # shear_factors: (batch_size, 6)
        self.shear_factors = shear_factors

    def get_shear_matrix(self, batch_size):
        shear_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        shear_matrix[:, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] = self.shear_factors
        return shear_matrix

    def __call__(self, point_cloud):
        batch_size = point_cloud.size(0)
        shear_matrix = self.get_shear_matrix(batch_size)
        return torch.bmm(point_cloud, shear_matrix)

        
class NonUniformScaling(object):
    def __init__(self, scale_factors):
        # scale_factors: (batch_size, 3)
        self.scale_factors = scale_factors

    def get_scaling_matrix(self, batch_size):
        scaling_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        scaling_matrix[:, [0, 1, 2], [0, 1, 2]] = self.scale_factors
        return scaling_matrix

    def __call__(self, point_cloud):
        batch_size = point_cloud.size(0)
        scaling_matrix = self.get_scaling_matrix(batch_size)
        return torch.bmm(point_cloud, scaling_matrix)

class ElasticDistortion(object):
    def __init__(self, sigma, alpha):
        # sigma and alpha should be of shape (batch_size, 1)
        self.sigma = sigma
        self.alpha = alpha

    def get_displacement_field(self, points):
        batch_size, num_points, dim = points.size()
        displacement = torch.randn(
            batch_size, num_points, dim, device=points.device
        ) * self.sigma.unsqueeze(2)
        displacement = displacement.permute(0, 2, 1)
        weight = torch.ones(dim, 1, 3, device=points.device) / 3
        displacement = F.conv1d(displacement, weight=weight, padding=1, groups=dim)
        displacement = displacement.permute(0, 2, 1)  
        return displacement

    def __call__(self, points):
        displacement = self.get_displacement_field(points)
        displacement *= self.alpha.unsqueeze(2)
        distorted_points = points + displacement
        return distorted_points

class SunisoidalWarping(object):
    def __init__(self, A, B):
        # A and B should be of shape (batch_size, 1)
        self.A = A
        self.B = B

    def get_warped_values(self, points):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        x_warped = x + self.A * torch.sin(self.B * y)
        y_warped = y + self.A * torch.sin(self.B * z)
        z_warped = z + self.A * torch.sin(self.B * x)
        return x_warped, y_warped, z_warped

    def __call__(self, points):
        x_warped, y_warped, z_warped = self.get_warped_values(points)
        warped_points = torch.stack((x_warped, y_warped, z_warped), dim=-1)
        return warped_points

class RadialDistortion(object):
    def __init__(self, k1, k2):
        # k1 and k2 should be of shape (batch_size, 1)
        self.k1 = k1
        self.k2 = k2

    def get_radial_displacement(self, points):
        r = torch.norm(points, dim=-1, keepdim=True)
        radial_displacement = self.k1 * r + self.k2 * r**3
        radial_displacement = radial_displacement * points / r
        return radial_displacement

    def __call__(self, points):
        radial_displacement = self.get_radial_displacement(points)
        distorted_points = points + radial_displacement
        return distorted_points

    
class Twisting(object):
    def __init__(self, k):
        # k should be of shape (batch_size, 1)
        self.k = k
    
    def get_twisted_values(self, points):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        x_twisted = x * torch.cos(self.k * z) - y * torch.sin(self.k * z)
        y_twisted = x * torch.sin(self.k * z) + y * torch.cos(self.k * z)
        z_twisted = z
        return x_twisted, y_twisted, z_twisted
    
    def __call__(self, points):
        x_twisted, y_twisted, z_twisted = self.get_twisted_values(points)
        twisted_points = torch.stack((x_twisted, y_twisted, z_twisted), dim=-1)
        return twisted_points
    