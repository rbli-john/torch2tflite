import torch
import torch.nn.functional


class ControlFlowModule(torch.nn.Module):
    def __init__(self):
        super(ControlFlowModule, self).__init__()
        self.l0 = torch.nn.Linear(4, 2)
        self.l1 = torch.nn.Linear(2, 1)

    def forward(self, input):
        if input.dim() > 1:
            return torch.tensor(0)

        out0 = self.l0(input)
        out0_relu = torch.nn.functional.relu(out0)
        return self.l1(out0_relu)


def run_trace():
    traced_module = torch.jit.trace(ControlFlowModule(), torch.randn(4))
    torch.jit.save(traced_module, 'output/controlflowmodule_traced.pt')
    loaded = torch.jit.load('output/controlflowmodule_traced.pt')
    print(loaded(torch.randn(2, 4)))


def run_script():
    scripted_module = torch.jit.script(ControlFlowModule())
    torch.jit.save(scripted_module, 'output/controlflowmodule_scripted.pt')
    loaded = torch.jit.load('output/controlflowmodule_scripted.pt')
    print(loaded(torch.randn(2, 4)))


def run_script_function():
    def test_sum(a, b):
        return a + b

    # Annotate the arguments to be int
    scripted_fn = torch.jit.script(test_sum, example_inputs=[(3, 4)])
    print(type(scripted_fn))  # torch.jit.ScriptFunction

    # See the compiled graph as Python code
    print(scripted_fn.code)

    # Call the function using the TorchScript interpreter
    print(scripted_fn(20, 100))


class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        # This parameter will be copied to the new ScriptModule
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        # When this submodule is used, it will be compiled
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input):
        output = self.weight.mv(input)

        # This calls the `forward` method of the `nn.Linear` module, which will
        # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
        output = self.linear(output)
        return output


def run_script_module():
    scripted_module = torch.jit.script(MyModule(2, 3))
    print(scripted_module.code)


def script_controlflow_module():
    scripted_module = torch.jit.script(ControlFlowModule())
    print(scripted_module.code)


from torch.utils.mobile_optimizer import optimize_for_mobile

if __name__ == '__main__':
    # run_trace()
    # run_script()
    # run_script_function()
    # run_script_module()
    script_controlflow_module()
