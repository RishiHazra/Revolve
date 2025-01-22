import json
import logging


def read_values(file_path):
    values = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.strip():  # Ensuring the line is not empty
                    values.append(float(line.strip()))
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} does not exist.")
    except ValueError as e:
        logging.error(f"Error parsing float: {e}")
    return values


def calculate_average_around_max(values, window=1000):
    if len(values) == 0:
        return None

    max_index = values.index(max(values))
    start_index = max(0, max_index - window)
    end_index = min(len(values), max_index + window + 1)
    average = sum(values[start_index:end_index]) / (end_index - start_index)
    return average, max_index


def return_score(file_path):
    values = read_values(file_path)
    average, max_index = calculate_average_around_max(values)
    return average
