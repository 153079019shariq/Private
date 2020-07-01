def myFun(*argv): 
	for arg in argv: 
		print (arg) 
	
myFun('Hello', 'Welcome', 'to', 'GeeksforGeeks')

################Output##################################
#Hello
#1Welcome
#to
#GeeksforGeeks


def myFun(**kwargs):  
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 
  
myFun(first ='Geeks', mid ='for', last='Geeks')     

#######################Output ##########################
#first == Geeks
#mid == for
#last == Geeks




