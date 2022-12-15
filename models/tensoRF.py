from .mixvoxels import *
from functools import partial

class TensorVMSplit(MixVoxels):
    def __init__(self, args, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(args, aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        self.dynamic_plane, self.dynamic_line = self.init_one_svd([16, 4, 4], self.gridSize, 0.1, device)

        self.static_density_plane, self.static_density_line = self.init_one_svd(self.density_n_comp, self.gridSize, self.args.voxel_init_static, device)
        self.static_app_plane, self.static_app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_static_mat = torch.nn.Linear(sum(self.app_n_comp), 27, bias=False).to(device)

        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp_dynamic, self.gridSize, self.args.voxel_init_dynamic, device)
        self.basis_den_mat = torch.nn.Linear(sum(self.density_n_comp_dynamic), self.den_dim, bias=False).to(device)

        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp_dynamic, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp_dynamic), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]), dtype=torch.float32)))  #
            line_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[vec_id], 1), dtype=torch.float32)))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_dynamic_init=0.02, lr_dynamic_basis=0.001):
        # dynamic_parameters
        grad_vars = [{'params': self.density_line, 'lr': lr_dynamic_init},
                     {'params': self.density_plane, 'lr': lr_dynamic_init},
                     {'params': self.app_line, 'lr': lr_dynamic_init},
                     {'params': self.app_plane, 'lr': lr_dynamic_init},
                     {'params': self.basis_mat.parameters(), 'lr':lr_dynamic_basis},
                     {'params': self.basis_den_mat.parameters(), 'lr': lr_dynamic_basis}
                     ]

        grad_vars += [{'params': self.renderDenModule.parameters(), 'lr': lr_dynamic_basis}]
        grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_dynamic_basis}]
        return grad_vars

    def get_static_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        # static parameters
        grad_vars = [{'params': self.static_density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.static_density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.static_app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.static_app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_static_mat.parameters(), 'lr': lr_init_network}
                    ]
        grad_vars += [{'params': self.renderStaticModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def get_dynamic_optparam_groups(self, lr = 0.02):
        grad_vars = []
        # parameters for dynamic prediction
        grad_vars += [{'params': self.dynamic_line, 'lr': lr}]
        grad_vars += [{'params': self.dynamic_plane, 'lr': lr}]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1_abstract(self, plane, line, reg=torch.abs):
        total = 0
        for idx in range(len(plane)):
            # total = total + torch.mean(reg(self.density_plane[idx])) + torch.mean(reg(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
            total = total + torch.mean(reg(plane[idx])) + torch.mean(reg(line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def density_L1(self, reg=torch.abs):
        return self.density_L1_abstract(self.density_plane, self.density_line, reg)

    def density_L1_static(self, reg=torch.abs):
        return self.density_L1_abstract(self.static_density_plane, self.static_density_line, reg)

    def TV_loss_abstract(self, reg, plane):
        total = 0
        for idx in range(len(plane)):
            total = total + reg(plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_dynamic(self, reg):
        return self.TV_loss_abstract(reg, self.dynamic_plane)

    def TV_loss_density(self, reg):
        return self.TV_loss_abstract(reg, self.density_plane)

    def TV_loss_static_density(self, reg):
        return self.TV_loss_abstract(reg, self.static_density_plane)

    def TV_loss_app(self, reg):
        return self.TV_loss_abstract(reg, self.app_plane)

    def TV_loss_static_app(self, reg):
        return self.TV_loss_abstract(reg, self.static_app_plane)

    def compute_single_value(self, xyz_sampled, plane, line):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(plane)):
            plane_coef_point = F.grid_sample(plane[idx_plane], coordinate_plane[[idx_plane]], mode=self.interpolation,
                                             align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(line[idx_plane], coordinate_line[[idx_plane]], mode=self.interpolation,
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            feature = feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
        return feature

    def compute_vector(self, xyz_sampled, plane, line, basis_mat):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(plane)):
            plane_coef_point.append(F.grid_sample(plane[idx_plane], coordinate_plane[[idx_plane]], mode=self.interpolation,
                                                  align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(line[idx_plane], coordinate_line[[idx_plane]], mode=self.interpolation,
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        feature = basis_mat((plane_coef_point * line_coef_point).T)

        return feature

    def compute_static_density(self, xyz_sampled):
        return self.compute_single_value(xyz_sampled, self.static_density_plane, self.static_density_line)

    def compute_static_app(self, xyz_sampled):
        return self.compute_vector(xyz_sampled, self.static_app_plane, self.static_app_line, self.basis_static_mat)

    def compute_dynamics(self, xyz_sampled):
        return self.compute_single_value(xyz_sampled, self.dynamic_plane, self.dynamic_line)

    def compute_densityfeature(self, xyz_sampled):
        return self.compute_vector(xyz_sampled, self.density_plane, self.density_line, self.basis_den_mat)

    def compute_appfeature(self, xyz_sampled):
        return self.compute_vector(xyz_sampled, self.app_plane, self.app_line, self.basis_mat)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.static_app_plane, self.static_app_line = self.up_sampling_VM(self.static_app_plane,
                                                                          self.static_app_line, res_target)
        self.static_density_plane, self.static_density_line = self.up_sampling_VM(self.static_density_plane,
                                                                                  self.static_density_line, res_target)

        self.dynamic_plane, self.dynamic_line = self.up_sampling_VM(self.dynamic_plane, self.dynamic_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.static_density_line[i] = torch.nn.Parameter(
                self.static_density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.static_app_line[i] = torch.nn.Parameter(
                self.static_app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.dynamic_line[i] = torch.nn.Parameter(
                self.dynamic_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.static_density_plane[i] = torch.nn.Parameter(
                self.static_density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.static_app_plane[i] = torch.nn.Parameter(
                self.static_app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.dynamic_plane[i] = torch.nn.Parameter(
                self.dynamic_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

