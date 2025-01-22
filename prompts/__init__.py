system_prompt = open("prompts/system_prompt", "r").read()
env_input_prompt = open("prompts/env_input", "r").read()
mutation_auto_prompt = open("prompts/mutation_auto", "r").read()
crossover_auto_prompt = open("prompts/crossover_auto", "r").read()
mutation_prompt = open("prompts/mutation", "r").read()
crossover_prompt = open("prompts/crossover", "r").read()

types = {
    "system_prompt": system_prompt,
    "env_input_prompt": env_input_prompt,
    "mutation_auto": mutation_auto_prompt,
    "crossover_auto": crossover_auto_prompt,
    "mutation": mutation_prompt,
    "crossover": crossover_prompt,
}
# print("system_prompt",system_prompt)
# print("env_input_prompt",env_input_prompt)
# print("mutation_auto_prompt",mutation_auto_prompt)
# print("crossover_auto_prompt",crossover_auto_prompt)
# print("mutation_prompt",mutation_prompt)
# print("crossover_prompt",crossover_prompt)
