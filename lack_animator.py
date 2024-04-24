# -*- coding: utf-8 -*-
'''Creates animations of hard sphere EDMD silumations from the output of lack_model.py'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def first_index_exceeding_sum(lst, threshold):
    
    '''Used for the timings of the animation'''
    
    cumulative_sum = 0
    for i, num in enumerate(lst):
        cumulative_sum += num
        if cumulative_sum > threshold:
            return i
        
    return len(lst) - 1  # Return the length of the list if threshold is not exceeded by any index



def update(frame):
    
    '''Function to update the animation'''
    
    # Check if frame exceeds total number of frames
    if frame >= total_frames:
        return im,  # No image to update
    
    # Calculate the index of the file corresponding to the current frame
    file_index = first_index_exceeding_sum(times, frame)
    if file_index == len(times):
        return im,  # No image to update
    
    # Load the PNG file corresponding to the current frame
    img = plt.imread(files[file_index])
    
    # Update the plot with the new image
    im.set_array(img)
    
    return im,



if __name__ == "__main__":
    
    # User inputs
    target_dir = '..\\Animation\\test2\\'  # Path to directory
    animation_time = 10  # Time in seconds for the total animation
    fps = 100 # Frames per second
    dpi = 600 # Resolution (dots per square inch)
    
    # Reading the timings file
    csv_file_name = target_dir + 'timings.csv'
    df = pd.read_csv(csv_file_name, sep=',')
    files = target_dir + df['files']
    
    # Calulate frames for each image
    total_frames = animation_time * fps
    times = (total_frames * df['times'] / sum(df['times'])).round().astype(int)
    
    # rounded_times = (total_frames * df['times'] / sum(df['times'])).round().astype(int)

    # # Compare the rounded values with the unrounded values
    # for unrounded, rounded in zip(total_frames * df['times'] / sum(df['times']), rounded_times):
    #     print(f"Unrounded: {unrounded}, Rounded: {rounded}")
    
    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # Load the first image
    img = plt.imread(files[0])
    im = ax.imshow(img)
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=sum(times), interval=1000 / fps) # Animating
    ani.save(target_dir + 'animation.mp4', writer='ffmpeg', fps=fps, dpi=dpi) # Save the animation
    plt.show() # Show the final frame of the animation
