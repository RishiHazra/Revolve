import json

def read_values(file_path):
    values = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Ensuring the line is not empty
                    values.append(float(line.strip()))
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except ValueError as e:
        print(f"Error parsing float: {e}")
    return values

def calculate_average_around_max(values, window=1000):
    if len(values) == 0:
        return None

    max_index = values.index(max(values))
    print("max index max score",max_index,values[max_index])
    start_index = max(0, max_index - window)
    end_index = min(len(values), max_index + window + 1)
    average = sum(values[start_index:end_index]) / (end_index - start_index)
    print("average",average)
    return average,max_index


def return_score(file_path):
    values = read_values(file_path)
    print("file path",file_path)
    average,max_index = calculate_average_around_max(values)
    print("Score at index:",average,max_index)
    return average 
