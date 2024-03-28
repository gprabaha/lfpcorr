import numpy as np
import h5py
import os

data_path = "/gpfs/milgram/project/chang/pg496/social_gaze_raw_mat"
input_fname = "Lynch_Cronenberg_ACCg_BLA_ACCg_01142019-acc.mat"
output_path = "./data"
output_fname = "reconstructed_signals.h5"
input_file_path = os.path.join(data_path, input_fname)
output_file_path = os.path.join(output_path, output_fname)

# Ensure the directory exists
os.makedirs(output_path, exist_ok=True)

# Define frequency bands (you may adjust these frequencies as needed)
# Example: 0-5 Hz, 5-10 Hz, 10-15 Hz
freq_bands = [(0, 5), (5, 10), (10, 15)]

def process_chunk(chunk):
    # Transpose the chunk
    chunk_transposed = chunk.T
    
    # Perform rfft on the transposed chunk
    chunk_fft = np.fft.rfft(chunk_transposed, axis=-1)
    
    # Reconstruct the signal at different frequency bands
    reconstructed_signals = []
    for band in freq_bands:
        low_freq, high_freq = band
        # Filter the Fourier coefficients
        filtered_fft = np.copy(chunk_fft)
        filtered_fft[:, high_freq:] = 0  # Zero out frequencies above the band
        if low_freq > 0:
            filtered_fft[:, :low_freq] = 0  # Zero out frequencies below the band
        # Reconstruct the signal
        reconstructed_signal = np.fft.irfft(filtered_fft, axis=-1)
        reconstructed_signals.append(reconstructed_signal)
    
    # reconstructed_signals will contain the reconstructed signals at different frequency bands
    # You can further process or analyze these signals as needed
    return reconstructed_signals

# Create or open the output HDF5 file
with h5py.File(output_file_path, 'w') as output_file:
    with h5py.File(input_file_path, 'r') as input_file:
        dataset = input_file['mat']  # Assuming 'mat' is the dataset name
        chunk_size = 1000  # Adjust the chunk size as needed
        
        # Create datasets in the output file to store reconstructed signals
        for i, band in enumerate(freq_bands):
            output_file.create_dataset(f'band_{i}', shape=(dataset.shape[1], 0), maxshape=(dataset.shape[1], None))
        
        # Determine the number of chunks
        num_chunks = dataset.shape[0] // chunk_size
        if dataset.shape[0] % chunk_size != 0:
            num_chunks += 1
        
        # Iterate over dataset chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, dataset.shape[0])
            chunk = dataset[start_idx:end_idx]
            reconstructed_signals = process_chunk(chunk)
            
            # Save reconstructed signals to the output file
            for i, signal in enumerate(reconstructed_signals):
                output_file[f'band_{i}'].resize((dataset.shape[1], output_file[f'band_{i}'].shape[1] + signal.shape[1]))
                output_file[f'band_{i}'][:, -signal.shape[1]:] = signal
