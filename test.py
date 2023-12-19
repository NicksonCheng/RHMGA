import torch
import torch.nn.functional as F
a=torch.rand(2, 3)
b=torch.tensor([[1.0,2.0,3.0],[1.0,2.0,3.0]])
c=torch.tensor([3.0,2.0,1.0])
d=(a*b).sum(dim=-1)

c=F.normalize(c,p=2,dim=-1)
print(c)