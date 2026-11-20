
from evalplus.data import get_human_eval_plus

problems = get_human_eval_plus()
problem = problems["HumanEval/0"]

print("Number of problems:", len(problems))
print("\nAvailable fields:", list(problem.keys()))
print("\nCoding prompt:")
print(problem["prompt"])
print("\nExpected function:")
print(problem["entry_point"])