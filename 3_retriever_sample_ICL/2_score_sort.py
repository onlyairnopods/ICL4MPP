import json

def sort(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    sorted_data = {}
    for k in data.keys():
        sorted_v = sorted(data[k], key=lambda x: x[-1], reverse=True)
        sorted_data[k] = sorted_v
    
    info_json = json.dumps(sorted_data, sort_keys=False, indent=4, separators=(',', ': '))
    with open(output_file, 'w') as f:
        f.write(info_json)

sort('./ClinTox_score.json', 'ClinTox_score_sort.json')