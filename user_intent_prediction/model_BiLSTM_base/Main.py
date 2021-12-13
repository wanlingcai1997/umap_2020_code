# Main.py

from Data import Data
from ReadConfiguration import ReadConfiguration
from ReadData import ReadData
from Initialization import Initialization
from Train import Train
from Evaluation import Evaluation
import time  


if __name__ == '__main__':
    
    # Step 1 Read Configuration
	print("===============================================================")
	start = time.process_time()
	ReadConfiguration.readConfiguration()
	end = time.process_time()
	print ('Step1: ReadConfiguration---- run time : %ss ' % str(end-start))
	print("===============================================================")


	# Step 2 Read Dataset
	print("===============================================================")	
	start = time.process_time()
	ReadData.readData()
	end = time.process_time()
	print ('Step2: ReadData---- run time : %ss ' % str(end-start))
	print("===============================================================")
    

	# Step 3 Initialization
	print("===============================================================")	
	start = time.process_time()
	Initialization.initialize()
	end = time.process_time()
	print ('Step3: Initialization---- run time : %ss ' % str(end-start))
	print("===============================================================")
    


	# Step 4 Train model
	print("===============================================================")
	start = time.process_time()
	Train.train()
	end = time.process_time()
	print ('Step4: Train model---- run time : %ss ' % str(end-start))
	print("===============================================================")

	# Step 5 Evaluation 
	print("===============================================================")
	start = time.process_time()
	Evaluation.evaluate()
	end = time.process_time()
	print ('Step 5 Evaluation---- run time : %ss ' % str(end-start))
	print("===============================================================")
	