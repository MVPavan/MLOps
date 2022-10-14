import logging

test_log = logging.getLogger(__name__)

Errors = {
    1: FileNotFoundError,
    2: AssertionError
}

def logAssert(expression,msg):
    if expression:return
    test_log.error(msg)
    assert expression, msg

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def exception_handler(func,):
    def _handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as err:
            raise err
    return _handler