import json


def read_json_lines(file_path):
    data = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.strip():  # Ensuring the line is not empty
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return data


def round_if_numeric(value, decimals=2):
    """Helper function to round values if they are numeric."""
    if isinstance(value, (float, int)):
        return round(value, decimals)
    elif isinstance(value, dict):
        return {k: round_if_numeric(v, decimals) for k, v in value.items()}
    elif isinstance(value, list):
        return [round_if_numeric(v, decimals) for v in value]
    return value


def serialize_dict(data, num_elements=25):
    if not data:
        return "No data available."

    # Assuming each entry in data is a dictionary
    keys = data[0].keys()  # Assuming all dictionaries have the same structure
    aggregated = {key: [round_if_numeric(d[key]) for d in data] for key in keys}

    ret_str = ""
    for key, values in aggregated.items():
        if isinstance(values, list) and len(values) > num_elements:
            step_size = len(values) / num_elements
            sampled_values = [values[int(i * step_size)] for i in range(num_elements)]
            sampled_values = [
                round_if_numeric(v) for v in sampled_values
            ]  # Round sampled values
            ret_str += f"{key}: {sampled_values}\n"
        else:
            # Ensure values are rounded, if numeric, before output
            rounded_values = [round_if_numeric(value) for value in values]
            ret_str += f"{key}: {rounded_values}\n"
    return ret_str


def return_history(file_path):
    data = read_json_lines(file_path)
    formatted_output = serialize_dict(data)
    return formatted_output


# test=return_history('episode_summary.json')
# print(test)
# in_context_samples_str = f'\n{test}'
# print(in_context_samples_str)
