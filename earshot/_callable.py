"""Make a module callable as its main function, so `earshot.describe(path)` works whether or not the submodule
`earshot.describe` has been imported (Python binds a submodule over a package attribute of the same name)."""
import sys
import types


def callable_module(name, fn_name=None):
    mod = sys.modules[name]
    fn = fn_name or name.rsplit('.', 1)[-1]

    class CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return getattr(self, fn)(*args, **kwargs)

    mod.__class__ = CallableModule
