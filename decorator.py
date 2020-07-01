


def decorator_function(orignal_function):
  def wrapper_function():
    print("wrapper_executed_this")
    orignal_function()
  return wrapper_function





def display():
  print("display_function_called")



decorated_display = decorator_function(display)


decorated_display()



################DECORATER$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

@decorator_function
def display():
  print("display_function_called")



#This  @ is equivalent to the following
display = decorator_function(display)        ##Check that the instance of the class(function) and argument to the class(function) should be same


display () 




# Now suppose we want to use decorator for two different function like "display" function above and  "display_info" function below. But note that the function "display_info" takes 2 argument. 
# So the Question arises how can I use the same same decorator for both of them ????



@decorator_function
def display_info(name,age):
   print(f"display_info_function ran with arguments name {name} age {age}")


 
display_info("John",25)     # This will give the following error TypeError: wrapper_function() takes 0 positional arguments but 2 were given

# So we have to change "wrapper_function()" to wrapper_function(*args) so that it can run for both "display()" and "display_info()"


