def calculate_fitness_score(log_file):
    total_fitness = 0
    num_episodes = 0

    with open(log_file, "r") as file:
        for line in file:
            if "Success=True" in line:
                # Extract the number of steps (removing the colon at the end)
                steps = int(line.split()[4].replace(":", ""))

                # Calculate fitness score using the formula y = ax + b
                a = -1 / 700
                b = 75 / 70
                fitness = a * steps + b
            else:
                # Success=False, fitness is 0
                fitness = 0

            total_fitness += fitness
            num_episodes += 1

    # Calculate the average fitness score
    if num_episodes > 0:
        average_fitness = total_fitness / num_episodes
    else:
        average_fitness = 0  # No episodes were found

    return average_fitness


# Example usage:
# average_fitness = calculate_fitness_score("performance_log.txt")
# print(f"Average Fitness Score: {average_fitness}")
