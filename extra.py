from typing import Any, Callable, TypeVar, cast
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)
import inspect
import functools
import sys
    
class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator"""

    def __call__(self, func: F) -> F:
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return func(*args, **kwargs)
        return cast(F, decorate_context)

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)

            # Generators are suspended and unsuspended at `yield`, hence we
            # make sure the grad mode is properly set every time the execution
            # flow returns into the wrapped generator and restored when it
            # returns through our `yield` to our caller (see PR #49017).
            try:
                # Issuing `None` to a generator fires it up
                with self.clone():
                    response = gen.send(None)

                while True:
                    try:
                        # Forward the response to our caller and get its next request
                        request = yield response

                    except GeneratorExit:
                        # Inform the still active generator about its imminent closure
                        with self.clone():
                            gen.close()
                        raise

                    except BaseException:
                        # Propagate the exception thrown at us by the caller
                        with self.clone():
                            response = gen.throw(*sys.exc_info())

                    else:
                        # Pass the last request to the generator and get its response
                        with self.clone():
                            response = gen.send(request)

            # We let the exceptions raised above by the generator's `.throw` or
            # `.send` methods bubble up to our caller, except for StopIteration
            except StopIteration as e:
                # The generator informed us that it is done: take whatever its
                # returned value (if any) was and indicate that we're done too
                # by returning it (see docs for python's return-statement).
                return e.value

        return generator_context

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()
class TrainMode(_DecoratorContextManager):
    def __init__(self, model,mode: bool):
        super().__init__()
        self.model = model
        self.mode = mode
    
    def clone(self):
        # override this method if your children class takes __init__ parameters
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