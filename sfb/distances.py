import torch
import geomloss


class EMDLoss(torch.nn.Module):
    """
    converges to Wasserstein distance for blur->0
    scaling < 0.4 for speed and scaling > 0.9 for accuracy
    """
    def __init__(self, p=2, **kwargs):
        super().__init__()
        blur = kwargs.get("blur", 1e-3)
        scal = kwargs.get("scaling", 0.9)
        self.reduce = kwargs.get("reduce", "mean")
        self.tensor = kwargs.get("tensor", False)
        # sollte beides True sein damit loss mehr Sinn macht
        debias = kwargs.get("debias", False)
        self.reweight = kwargs.get("reweight", False)
        if p == 2:
            self.tensor = True
            self.op = geomloss.SamplesLoss(
                loss='sinkhorn', debias=debias, p=1, blur=blur, scaling=scal, backend="tensorized"
            )
        elif p == 1:
            self.op = geomloss.SamplesLoss(
                loss='sinkhorn', debias=debias, blur=blur, scaling=scal, backend="online", cost="Sum(Abs(X-Y))"
            )
        elif p == .75:
            self.op = geomloss.SamplesLoss(
                loss='sinkhorn',
                debias=debias,
                blur=blur,
                scaling=scal,
                backend="online",
                cost="Sum(Powf(Abs(X-Y), IntInv(2)))"
            )
        else:
            self.op = geomloss.SamplesLoss(
                loss='sinkhorn',
                debias=debias,
                blur=blur,
                scaling=scal,
                backend="online",
                cost="Sum(Pow(Abs(X-Y),{}))".format(int(p))
            )

    def forward(self, x, y):
        if self.reweight:
            N = max(x.size(1), y.size(1))  # should not really matter if max/min
            if x.size(0) > 1 and self.tensor:
                a = torch.ones(x.size(0), x.size(1)).type_as(x) / N
                b = torch.ones(y.size(0), y.size(1)).type_as(y) / N
            else:
                a = torch.ones(x.size(1)).type_as(x) / N
                b = torch.ones(y.size(1)).type_as(y) / N
            args = [a, x, b, y]
        else:
            args = [x, y]

        if x.size(0) > 1 and self.tensor:
            z = self.op(*args)
        else:
            z = []
            for k in range(x.size(0)):
                # z.append(self.op(x[k:k + 1], y[k:k + 1]))
                z.append(self.op(*[a[k:k + 1] for a in args]))
            z = torch.cat(z)

        if self.reduce is None:
            return z
        elif self.reduce == "sum":
            return z.sum()
        else:
            return z.mean()
