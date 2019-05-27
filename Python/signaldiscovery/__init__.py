try:
   
	import os
	os.environ["KERAS_BACKEND"] = 'tensorflow'

	    
except ImportError:
    print ("Signal Discovery init error - Error importing modules")
