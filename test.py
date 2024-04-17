from tqdm import tqdm
import time

# Set total number of iterations
total_iterations = 100

# Initialize tqdm with the total number of iterations
progress_bar = tqdm(total=total_iterations, unit="iteration")

# Iterate over each iteration
for i in range(total_iterations):
    # Perform some task (simulation)
    time.sleep(0.1)
    # Update progress bar
    progress_bar.update(1)
    # Print something at the end of each iteration
    progress_bar.set_postfix({"Message": "Task completed!"}, refresh=)

# Close progress bar
progress_bar.close()
