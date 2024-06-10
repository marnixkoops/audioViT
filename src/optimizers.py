import torch


class Adan(torch.optim.Optimizer):
    """
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models
    https://arxiv.org/pdf/2208.06677
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.02, 0.08, 0.01),
        eps=1e-8,
        weight_decay=0,
        restart_cond: callable = None,
    ):

        assert len(betas) == 3

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            restart_cond=restart_cond,
        )

        super().__init__(params, defaults)

    def exists(self, val):
        return val is not None

    def step(self, closure=None):
        loss = None

        if self.exists(closure):
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            restart_cond = group["restart_cond"]

            for p in group["params"]:
                if not self.exists(p.grad):
                    continue

                data, grad = p.data, p.grad.data
                assert not grad.is_sparse

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = torch.zeros_like(grad)
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                    state["n"] = torch.zeros_like(grad)

                step, m, v, n, prev_grad = (
                    state["step"],
                    state["m"],
                    state["v"],
                    state["n"],
                    state["prev_grad"],
                )

                if step > 0:
                    prev_grad = state["prev_grad"]

                    # main algorithm
                    m.mul_(1 - beta1).add_(grad, alpha=beta1)
                    grad_diff = grad - prev_grad
                    v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)
                    next_n = (grad + (1 - beta2) * grad_diff) ** 2
                    n.mul_(1 - beta3).add_(next_n, alpha=beta3)

                # bias correction terms
                step += 1
                correct_m, correct_v, correct_n = map(
                    lambda n: 1 / (1 - (1 - n) ** step), (beta1, beta2, beta3)
                )

                # gradient step
                def grad_step_(data, m, v, n):
                    weighted_step_size = lr / (n * correct_n).sqrt().add_(eps)

                    denom = 1 + weight_decay * lr

                    data.addcmul_(
                        weighted_step_size,
                        (m * correct_m + (1 - beta2) * v * correct_v),
                        value=-1.0,
                    ).div_(denom)

                grad_step_(data, m, v, n)

                # restart condition
                if self.exists(restart_cond) and restart_cond(state):
                    m.data.copy_(grad)
                    v.zero_()
                    n.data.copy_(grad**2)

                    grad_step_(data, m, v, n)

                # set new incremented step
                prev_grad.copy_(grad)
                state["step"] = step

        return loss


class Nadam(torch.optim.Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).

    - Sources: "Incorporating Nesterov Momentum into Adam"
        https://cs229.stanford.edu/proj2015/054_report.pdf
        http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)


        Originally taken from: https://github.com/pytorch/pytorch/pull/1408
        NOTE: Has potential issues but does work well on some problems.
    """

    def __init__(
        self,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        schedule_decay=4e-3,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            schedule_decay=schedule_decay,
        )
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m_schedule"] = 1.0
                    state["exp_avg"] = grad.new().resize_as_(grad).zero_()
                    state["exp_avg_sq"] = grad.new().resize_as_(grad).zero_()

                # Warming momentum schedule
                m_schedule = state["m_schedule"]
                schedule_decay = group["schedule_decay"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                state["step"] += 1
                t = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                momentum_cache_t = beta1 * (1.0 - 0.5 * (0.96 ** (t * schedule_decay)))
                momentum_cache_t_1 = beta1 * (
                    1.0 - 0.5 * (0.96 ** ((t + 1) * schedule_decay))
                )
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                state["m_schedule"] = m_schedule_new

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                exp_avg_sq_prime = exp_avg_sq / (1.0 - beta2**t)
                denom = exp_avg_sq_prime.sqrt_().add_(eps)

                p.data.addcdiv_(
                    -group["lr"] * (1.0 - momentum_cache_t) / (1.0 - m_schedule_new),
                    grad,
                    denom,
                )
                p.data.addcdiv_(
                    -group["lr"] * momentum_cache_t_1 / (1.0 - m_schedule_next),
                    exp_avg,
                    denom,
                )

        return loss


class NvidiaNovoGrad(torch.optim.Optimizer):
    """
    Nvidia NovoGrad Optimizer.

    Original implementation by Nvidia from Jasper example:
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper

    Paper: Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of
    Deep Networks https://arxiv.org/abs/1905.11286


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.95, 0.98),
        eps=1e-8,
        weight_decay=0,
        grad_averaging=False,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            amsgrad=amsgrad,
        )

        super(NvidiaNovoGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NvidiaNovoGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros([])
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros([])

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                grad.div_(denom)
                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)
                if group["grad_averaging"]:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(-group["lr"], exp_avg)

        return loss
