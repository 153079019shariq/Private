class decorator_class(object):
  def __init__(self,orignal_function):
    self.orig_func  = orignal_function

  def __call__(self,*args,**kwargs):
    print(f"call method executed this before {self.orig_func.__name__}")
    self.orig_func(*args,**kwargs)





@decorator_class
def display():
  print("display_function ran")


@decorator_class
def display_info(name,age):
  print(f"display_info ran with argument name {name} age {age}")



display_info("John",26)
display()


#########################OUTPUT##################################

# call method executed this before display_info
# display_info ran with argument name John age 26
# call method executed this before display
# display_function ran



