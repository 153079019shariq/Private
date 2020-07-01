
def logger(func):
  
  def log_func(*args):
    print(f"Running function {func} with arguments {args}")
    print(func(*args))
  return log_func



def add(x,y):
  return x+y


def sub(x,y):
  return x-y


def mul(x,y):
  return x*y



## WHEN WE PASS AN ARGUMENT TO THE INSTANCE OF THE OBJECT IT GOES as the argument of INNER FUNCTION i.e log_func

check1 = logger(add)
print(check1.__name__)     #  it will print the inner function name log_func and not outer function logger
check1(2,3)                ## Output   Running function <function add at 0x7ff8c0f5d510> with arguments (2, 3)
                           ## 5





check2 = logger(sub)
print(check2.__name__)    
check2(5,4)              ##  it will print the inner function name log_func and not outer function logger
                         ##  Running function <function sub at 0x7fa796a7dae8> with arguments (5, 4)
                         ##  1
 

check3 = logger(mul)
print(check3.__name__)   ## it will print the inner function name log_func and not outer function logger
check3(2,5)              ## Running function <function mul at 0x7f9c8a0a6b70> with arguments (2, 5)
                         ## 10

