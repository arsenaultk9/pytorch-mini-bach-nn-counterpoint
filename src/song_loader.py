import json
 
def load_song(file_path):
    json_file = open(file_path)
    data = json.load(json_file)
    json_file.close()

    return data