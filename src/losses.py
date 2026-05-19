import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation (channel-wise, unweighted)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, C, H, W) one-hot encoded ground truth
        """
        pred = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]

        total_loss = 0.0
        for c in range(num_classes):
            p = pred[:, c].flatten(1)
            t = target[:, c].flatten(1)
            intersection = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_loss += 1.0 - dice.mean()

        return total_loss / num_classes


class PDELoss(nn.Module):
    """
    PDE loss enforcing Fisher-KPP reaction-diffusion equation on
    estimated tumour cell density.

    PDE: du/dt = D * laplacian(u) + rho * u * (1 - u)

    2D Laplacian approximated with discrete kernel:
        [[0, 1, 0],
         [1,-4, 1],
         [0, 1, 0]]
    """

    def __init__(self, d_range=(0.02, 1.5), rho_range=(0.002, 0.2)):
        super().__init__()
        self.d_range = d_range
        self.rho_range = rho_range

        # 2D Laplacian kernel (4-connectivity)
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]]
        ).reshape(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)

    def forward(self, u_hat, u_hat_prev=None, dt=1.0):
        """
        Args:
            u_hat: (B, 1, H, W) current density estimate
            u_hat_prev: (B, 1, H, W) previous time step density (if None, enforce steady-state)
            dt: time step size
        Returns:
            pde_loss: scalar
        """
        B = u_hat.shape[0]
        device = u_hat.device

        # Sample random biophysical parameters per batch
        d = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.d_range)
        rho = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.rho_range)

        # Compute Laplacian via convolution
        laplacian_u = F.conv2d(u_hat, self.laplacian_kernel, padding=1)
        diffusion = d * laplacian_u

        # Reaction term (logistic growth)
        reaction = rho * u_hat * (1.0 - u_hat)

        # Time derivative approximation
        if u_hat_prev is not None:
            du_dt = (u_hat - u_hat_prev) / dt
        else:
            # When no previous step available, enforce steady-state (du/dt ~ 0)
            du_dt = torch.zeros_like(u_hat)

        # PDE residual: du/dt - D*laplacian(u) - rho*u*(1-u) = 0
        residual = du_dt - diffusion - reaction

        # MSE of residual
        pde_loss = (residual ** 2).mean()

        return pde_loss


class BoundaryConditionLoss(nn.Module):
    """
    Neumann (zero-flux) boundary condition loss.
    Enforces: D * (du/dn) = 0 on domain boundaries.
    In 2D, this means zero gradient at all 4 edges.
    """

    def __init__(self):
        super().__init__()

    def forward(self, u_hat):
        """
        Args:
            u_hat: (B, 1, H, W) density estimate
        Returns:
            bc_loss: scalar
        """
        # Top edge: du/dy at y=0
        grad_top = u_hat[:, :, 1, :] - u_hat[:, :, 0, :]
        # Bottom edge: du/dy at y=H-1
        grad_bottom = u_hat[:, :, -1, :] - u_hat[:, :, -2, :]
        # Left edge: du/dx at x=0
        grad_left = u_hat[:, :, :, 1] - u_hat[:, :, :, 0]
        # Right edge: du/dx at x=W-1
        grad_right = u_hat[:, :, :, -1] - u_hat[:, :, :, -2]

        bc_loss = (
            (grad_top ** 2).mean() +
            (grad_bottom ** 2).mean() +
            (grad_left ** 2).mean() +
            (grad_right ** 2).mean()
        )

        return bc_loss


class BiophysicsInformedLoss(nn.Module):
    """Combined loss: L_total = L_dice + lambda1 * L_pde + lambda2 * L_bc"""

    def __init__(self, lambda_pde=1.0, lambda_bc=1.0, d_range=(0.02, 1.5), rho_range=(0.002, 0.2)):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.pde_loss = PDELoss(d_range=d_range, rho_range=rho_range)
        self.bc_loss = BoundaryConditionLoss()
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc

    def forward(self, pred, target, u_hat):
        """
        Args:
            pred: (B, C, H, W) segmentation logits
            target: (B, C, H, W) one-hot ground truth
            u_hat: (B, 1, H, W) estimated tumour cell density
        Returns:
            total_loss, dict of individual losses
        """
        l_dice = self.dice_loss(pred, target)
        l_pde = self.pde_loss(u_hat)
        l_bc = self.bc_loss(u_hat)

        total = l_dice + self.lambda_pde * l_pde + self.lambda_bc * l_bc

        loss_dict = {
            "dice": l_dice.item(),
            "pde": l_pde.item(),
            "bc": l_bc.item(),
            "total": total.item(),
        }

        return total, loss_dict
