import copy
import json

def store_result_json_file(file_name,data):
    path = 'result_analysis/'+file_name
    with open(path, 'w') as newfile:
        json.dump(data, newfile, indent=4)
    return 
