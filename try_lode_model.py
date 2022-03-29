import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def foo(x=...):
    return x


def run2():
    # l = nn.Linear(2, 2)
    l1 = nn.Linear(2, 2)
    l2 = nn.Linear(2, 2)
    net = nn.Sequential(l1, l2)
    for idx, m in enumerate(net.named_modules()):
        print(idx, '->', m)
    # print(net)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv = nn.Conv2d(3, 10, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(10 * 15 * 15, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        # print('x.size after pool', x.size())
        x = x.view(-1, 10 * 15 * 15)
        x = self.fc(x)
        return x


def test_my_module():
    my_module = MyModule()
    my_module.eval()
    input = torch.randn(1, 3, 30, 30)
    out_np = my_module(input).detach().numpy()
    print('out_shape=', out_np.shape)
    print(out_np)


def run3():
    my_module = MyModule()
    for idx, m in enumerate(my_module.named_modules()):
        print(idx, '->', m)


def run4():
    large = torch.arange(1, 1000)
    small = large[0:5]
    print(type(small))
    print(small.storage().size())

    small_copy = small.clone()
    print(small_copy.storage().size())


def run5():
    bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
    print(list(bn.named_parameters()))


def load_model():
    device = torch.device("cpu")
    model = torch.load('tests/mobilenetv2_model.pt', map_location=device)
    # summary(model, (3, 224, 224), device=device)
    print(type(model))


def save_my_module():
    my_module = MyModule()
    torch.save(my_module, 'tests/my_module.pt')


def save_traced_module():
    my_module = MyModule()
    traced_module = torch.jit.trace(my_module, torch.randn(1, 3, 30, 30))
    torch.jit.save(traced_module, 'tests/my_module_traced.pt')


def load_traced_module():
    traced_module = torch.jit.load('tests/my_module_traced.pt', map_location='cpu')
    print(type(traced_module))


def save_script_module():
    my_module = MyModule()
    script_module = torch.jit.script(my_module)
    torch.jit.save(script_module, 'tests/my_module_script.pt')

    # verify
    loaded_model = torch.jit.load('tests/my_module_script.pt')
    print(loaded_model.code)


def load_script_module():
    script_module = torch.jit.load('tests/my_module_script.pt', map_location='cpu')
    print(type(script_module))



if __name__ == '__main__':
    # test_my_module()
    # load_model()
    # save_my_module()
    # save_traced_module()
    # load_traced_module()
    # save_script_module()
    load_script_module()


