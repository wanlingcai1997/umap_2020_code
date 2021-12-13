# Main.py

from Data import Data
from ReadConfiguration import ReadConfiguration
from ReadData import ReadData
from ConstructFeature import ConstructFeature
from Train import Train
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
    

	# Step 3 Construct Feature
	print("===============================================================")	
	start = time.process_time()
	ConstructFeature.constructFeature()
	end = time.process_time()
	print ('Step3: ConstructFeature---- run time : %ss ' % str(end-start))
	print("===============================================================")
    


	# Step 4 Train model & Test
	print("===============================================================")
	start = time.process_time()
	Train.train()
	end = time.process_time()
	print ('Step4: Train model---- run time : %ss ' % str(end-start))
	print("===============================================================")
	
