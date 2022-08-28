from typing import Any, Callable, TypeVar, cast
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)
import inspect
import functools
import sys
    
class _DecoratorContextManager:
    def __call__(self, func: F) -> F:
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return func(*args, **kwargs)
        return cast(F, decorate_context)

    def _wrap_generator(self, func):
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)
            try:
                with self.clone():
                    response = gen.send(None)

                while True:
                    try:
                        request = yield response
                    except GeneratorExit:
                        with self.clone():
                            gen.close()
                        raise
                    except BaseException:
                        with self.clone():
                            response = gen.throw(*sys.exc_info())
                    else:
                        with self.clone():
                            response = gen.send(request)

            except StopIteration as e:
                return e.value

        return generator_context

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        return self.__class__()
class TrainMode(_DecoratorContextManager):
    def __init__(self, model,mode: bool):
        super().__init__()
        self.model = model
        self.mode = mode
    
    def clone(self):
        return self.__class__(self.model, self.mode)
    
    def __enter__(self):
        self.mode_bak = self.model.training
        self.model.train(self.mode)
        return self.model
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.model.train(self.mode_bak)
        return None

# """usage
# """
# net = get_net()
# with TrainMode(model=net, mode=False) as netb:
#     print(netb.training)
# print(net.training)

# @TrainMode(model=net, mode=False)
# def test():
#     print(net.training)

# test()
# print(net.training)
