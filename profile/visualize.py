import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the similarities from the JSON file
with open('/workspace/repo/flux/output/similarities.json', 'r') as f:
    all_similarities = json.load(f)

# Initialize dictionaries to accumulate similarities per timestep
double_img_sims_per_timestep = {}
double_txt_sims_per_timestep = {}
single_img_sims_per_timestep = {}
count_per_timestep = {}

# Iterate over all collected similarities
for data in all_similarities:
    timestep = data['timestep']
    
    # Convert lists back to NumPy arrays
    double_img_sims = np.array(data['double_block_img_sims'])
    double_txt_sims = np.array(data['double_block_txt_sims'])
    single_img_sims = np.array(data['single_block_img_sims'])
    
    # Initialize accumulators if not already done
    if timestep not in double_img_sims_per_timestep:
        double_img_sims_per_timestep[timestep] = np.zeros_like(double_img_sims)
        double_txt_sims_per_timestep[timestep] = np.zeros_like(double_txt_sims)
        single_img_sims_per_timestep[timestep] = np.zeros_like(single_img_sims)
        count_per_timestep[timestep] = 0
    
    # Accumulate the similarities
    double_img_sims_per_timestep[timestep] += double_img_sims
    double_txt_sims_per_timestep[timestep] += double_txt_sims
    single_img_sims_per_timestep[timestep] += single_img_sims
    count_per_timestep[timestep] += 1

# Initialize dictionaries to store average similarities per timestep
avg_double_img_sims_per_timestep = {}
avg_double_txt_sims_per_timestep = {}
avg_single_img_sims_per_timestep = {}

# Compute the averages for each timestep
for timestep in sorted(count_per_timestep.keys()):
    count = count_per_timestep[timestep]
    print(f"Timestep: {timestep}, Count: {count}")

    avg_double_img_sims_per_timestep[timestep] = double_img_sims_per_timestep[timestep] / count
    avg_double_txt_sims_per_timestep[timestep] = double_txt_sims_per_timestep[timestep] / count
    avg_single_img_sims_per_timestep[timestep] = single_img_sims_per_timestep[timestep] / count


print("Sanity check")
for timestep in sorted(count_per_timestep.keys()):
    print(avg_double_img_sims_per_timestep[timestep].shape)
    print(avg_double_txt_sims_per_timestep[timestep].shape)
    print(avg_single_img_sims_per_timestep[timestep].shape)


full_double_img_sims = avg_double_img_sims_per_timestep[0]
full_double_txt_sims = avg_double_txt_sims_per_timestep[0]
full_single_img_sims = avg_single_img_sims_per_timestep[0]

for timestep in sorted(count_per_timestep.keys()):
    if timestep == 0:
        continue
    full_double_img_sims += avg_double_img_sims_per_timestep[timestep]
    full_double_txt_sims += avg_double_txt_sims_per_timestep[timestep]
    full_single_img_sims += avg_single_img_sims_per_timestep[timestep]

score_double_img = [full_double_img_sims[x][x+1] for x in range(full_double_img_sims.shape[0]-1)]
score_double_txt = [full_double_txt_sims[x][x+1] for x in range(full_double_txt_sims.shape[0]-1)]
score_single_img = [full_single_img_sims[x][x+1] for x in range(full_single_img_sims.shape[0]-1)]

sorted_double_img = sorted(range(len(score_double_img)), key=lambda i: score_double_img[i], reverse=True)
sorted_double_txt = sorted(range(len(score_double_txt)), key=lambda i: score_double_txt[i], reverse=True)
sorted_single_img = sorted(range(len(score_single_img)), key=lambda i: score_single_img[i], reverse=True)

print("Double_img: ", sorted_double_img)
print("Double_txt: ", sorted_double_txt)
print("Single_img: ", sorted_single_img)

# Create a directory to save the plots if it doesn't exist
output_dir = 'average_similarity_maps_per_timestep'
os.makedirs(output_dir, exist_ok=True)

# Sort timesteps for consistent ordering
sorted_timesteps = sorted(count_per_timestep.keys())

for timestep in sorted_timesteps:
    # Average DoubleStreamBlocks Image Similarities
    avg_double_img_sims = avg_double_img_sims_per_timestep[timestep]
    plt.figure(figsize=(8, 6))
    # sns.heatmap(avg_double_img_sims, annot=False, fmt=".2f", cmap='viridis')
    sns.heatmap(avg_double_img_sims, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title(f'Average DoubleStreamBlocks Image Similarities\nTimestep: {timestep}')
    plt.xlabel('Block Index')
    plt.ylabel('Block Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_double_img_sims_timestep_{timestep}.png'))
    plt.close()
    
    # Average DoubleStreamBlocks Text Similarities
    avg_double_txt_sims = avg_double_txt_sims_per_timestep[timestep]
    plt.figure(figsize=(8, 6))
    # sns.heatmap(avg_double_txt_sims, annot=False, fmt=".2f", cmap='magma')
    sns.heatmap(avg_double_txt_sims, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title(f'Average DoubleStreamBlocks Text Similarities\nTimestep: {timestep}')
    plt.xlabel('Block Index')
    plt.ylabel('Block Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_double_txt_sims_timestep_{timestep}.png'))
    plt.close()
    
    # Average SingleStreamBlocks Image Similarities
    avg_single_img_sims = avg_single_img_sims_per_timestep[timestep]
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_single_img_sims, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title(f'Average SingleStreamBlocks Image Similarities\nTimestep: {timestep}')
    plt.xlabel('Block Index')
    plt.ylabel('Block Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_single_img_sims_timestep_{timestep}.png'))
    plt.close()
