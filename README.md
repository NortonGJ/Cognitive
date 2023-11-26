# Cognitive
Main theme of the project:
Get real-live noise situation in SW/USW, add some synthetic noise, prepair data for ML. 

getData - get real-live noise situation in SW/USW via connected RSP-1 device, save it in 30 sec samples (0.1 sec in one np.ndarray line)

addNoise - adding synthetic noise, prepairing data for ML (saving: preShuffle data file, shuffled data file, label files)

New_Spectr - script to show existing data in dynamic
(TODO: add prediction display, add multiple ways of getting predictions (algorithmic, NN))

NN_test - notebook with example of NN learning process
