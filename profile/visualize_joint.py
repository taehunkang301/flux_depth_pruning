import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the similarities from the JSON file
#with open('/workspace/repo/flux/output/similarities_dev_joint_d5_100.json', 'r') as f:
with open('/workspace/repo/flux/output/similarities_dev_joint_100.json', 'r') as f:
    all_similarities = json.load(f)

# Initialize dictionaries to accumulate similarities per timestep
double_img_sims_per_timestep = {}
double_txt_sims_per_timestep = {}
single_img_sims_per_timestep = {}

double_block_sims_per_timestep = {}
single_block_sims_per_timestep = {}
count_per_timestep = {}

# Iterate over all collected similarities
for data in all_similarities:
    timestep = data['timestep']
    
    # Convert lists back to NumPy arrays
    # double_img_sims = np.array(data['double_block_img_sims'])
    # double_txt_sims = np.array(data['double_block_txt_sims'])
    # single_img_sims = np.array(data['single_block_img_sims'])

    double_block_sims = np.array(data['double_block_sims'])
    single_block_sims = np.array(data['single_block_sims'])
    
    # Initialize accumulators if not already done
    if timestep not in double_img_sims_per_timestep:
        double_block_sims_per_timestep[timestep] = np.zeros_like(double_block_sims)
        single_block_sims_per_timestep[timestep] = np.zeros_like(single_block_sims)
        count_per_timestep[timestep] = 0
    
    # Accumulate the similarities
    double_block_sims_per_timestep[timestep] += double_block_sims
    single_block_sims_per_timestep[timestep] += single_block_sims
    count_per_timestep[timestep] += 1

# Initialize dictionaries to store average similarities per timestep
avg_double_sims_per_timestep = {}
avg_single_sims_per_timestep = {}

# Compute the averages for each timestep
for timestep in sorted(count_per_timestep.keys()):
    count = count_per_timestep[timestep]
    print(f"Timestep: {timestep}, Count: {count}")

    avg_double_sims_per_timestep[timestep] = double_block_sims_per_timestep[timestep] / count
    avg_single_sims_per_timestep[timestep] = single_block_sims_per_timestep[timestep] / count

print("Sanity check")
for timestep in sorted(count_per_timestep.keys()):
    print(avg_double_sims_per_timestep[timestep].shape)
    print(avg_single_sims_per_timestep[timestep].shape)

full_double_sims = avg_double_sims_per_timestep[0]
full_single_sims = avg_single_sims_per_timestep[0]

for timestep in sorted(count_per_timestep.keys()):
    if timestep == 0:
        continue
    full_double_sims += avg_double_sims_per_timestep[timestep]
    full_single_sims += avg_single_sims_per_timestep[timestep]

score_double_sim = [full_double_sims[x][x+1] for x in range(full_double_sims.shape[0]-1)]
score_single_sim = [full_single_sims[x][x+1] for x in range(full_single_sims.shape[0]-1)]

sorted_double_sim = sorted(range(len(score_double_sim)), key=lambda i: score_double_sim[i], reverse=True)
sorted_single_sim = sorted(range(len(score_single_sim)), key=lambda i: score_single_sim[i], reverse=True)

print("Double sim: ", sorted_double_sim)
print("Single sim: ", sorted_single_sim)

def get_sorted_index_list(double, single):
    # Create a combined list of tuples, each tuple will hold the value, index and the prefix ('D' for double, 'S' for single)
    combined = [('D'+str(i), value) for i, value in enumerate(double)] + [('S'+str(i), value) for i, value in enumerate(single)]
    
    # Sort this combined list by the value in descending order
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    
    # Extract only the prefixed indexes in order
    sorted_indexes = [item[0] for item in sorted_combined]
    
    return sorted_indexes


print("Whole sim: ", get_sorted_index_list(score_double_sim, score_single_sim))

raise


# Create a directory to save the plots if it doesn't exist
output_dir = 'average_similarity_maps_dev_100'
os.makedirs(output_dir, exist_ok=True)

# Sort timesteps for consistent ordering
sorted_timesteps = sorted(count_per_timestep.keys())

for timestep in sorted_timesteps:
    # Average DoubleStreamBlocks Similarities
    avg_double_sims = avg_double_sims_per_timestep[timestep]
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_double_sims, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title(f'Average DoubleStreamBlocks Similarities\nTimestep: {timestep}')
    plt.xlabel('Block Index')
    plt.ylabel('Block Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_double_sims_timestep_{timestep}.png'))
    plt.close()

    # Average SingleStreamBlocks Image Similarities
    avg_single_sims = avg_single_sims_per_timestep[timestep]
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_single_sims, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title(f'Average SingleStreamBlocks Similarities\nTimestep: {timestep}')
    plt.xlabel('Block Index')
    plt.ylabel('Block Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_single_sims_timestep_{timestep}.png'))
    plt.close()

