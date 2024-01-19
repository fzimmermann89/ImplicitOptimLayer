class DifferentiableSolver(torch.autograd.Function):
    """
    Wrapper around optim.bfgs to make it differentiable
    
    Solves argmin_x ||f(x)-y ||^2 + l*||x-xreg||^2 with BFGS in the forward pass using xreg or x0 as initial point
    and calculates the gradients wrt y, l and xreg in the backward pass
    """

    @staticmethod
    def forward(ctx, f: Callable, y: Tensor, xreg: Tensor, l: Tensor, x0: Tensor|None):
        xreg = xreg.detach()
        if x0 is None:
	        x = xreg.clone().requires_grad_(True) # Use xreg as initial point for regresison
        else:
            x = x0.clone().requires_grad_(True) # Use x0 as initial point for regresison
        y = y.detach()
        lscalar = l.item()

        # The objective functional ||f(x)-y ||^2 + l*||x-xreg||^2
        objective_function = lambda x, y, xreg, l: torch.nn.functional.mse_loss(f(x), y, reduction="sum") + l * torch.nn.functional.mse_loss(x, xreg, reduction="sum")

        
        # Solve the inner problem, this could be any solver, we use L-BFGS here
        optim = torch.optim.LBFGS([x], history_size=10, max_iter=50, tolerance_change=1e-12, tolerance_grad=1e-12, line_search_fn="strong_wolfe")
        
        # Run BFGS
        def closure():
            x.grad = None
            objective = objective_function(x, y, xreg, lscalar)
            objective.backward()
            return objective
        optim.step(closure)
        
        x.grad = None
        ctx.save_for_backward(x, y, xreg, l)
        ctx.f = f
        ctx.objective_function = objective_function
        return x

    @staticmethod
    def backward(ctx, grad):
        xprime, y, xreg, l = ctx.saved_tensors
        xprime = xprime.detach().clone().requires_grad_(True)
        

        # Only wrt parameters that need a gradient will later on be differentiated and need to have requires_grad_(True)
        params = (y, xreg, l)
        params = [p.detach().clone().requires_grad_(True) if ctx.needs_input_grad[i + 1] else p.detach() for i, p in enumerate(params)]
        dparams = [p for p in params if p.requires_grad]  # Parameters that need a gradient
        (y, xreg, l) = params

        # objective function
        objective = lambda x: ctx.objective_function(x, y, xreg, l)

        A = lambda v: torch.autograd.functional.vhp(objective, xprime, v=v)[1]  # Hessian Operator
        g = cg(A, grad, maxiter=50, tol=1e-6, dims=tuple(range(-1, -xreg.ndim - 1, -1)))  # Solve H*iHv=grad

        with torch.enable_grad():  # Mixed gradients of objective
            dobj_dprime = torch.autograd.grad(objective(xprime), xprime, create_graph=True)[0] 
            ddobj_dprimed_dparam = list(torch.autograd.grad(dobj_dprime, dparams, -g))
        
        grad_output = [None]  # no gradient for f
        for need_grad in ctx.needs_input_grad[1:4]:
            grad_output.append(ddobj_dprimed_dparam.pop(0))
        else:
            grad_output.append(None)

        return tuple(grad_output)
