# speaker-cluster
BIC speaker cluster for diarisation

This code is a simple BIC clusterer for speaker diarisation. Clustering based on Euclidean distances and Arithmetic Harmonic Sphericity.

$ python2 speakercluster.py 
usage: speakercluster.py [-h] [-d {ahs,bic,euclid,gish}] [-c {diagonal,full}]
                         [-p PENALTY] [-t THRESHOLD] [-m MAX_CLUSTERS]
                         [-s SHORT_TIME] [-f FOLDER] [-r REFERENCE] [--trace]
                         [--collar COLLAR]
                         scp
                         
                         
                         
Requires an scp file pointing to segments in a HTK feature file:

CORPUS-FILE1_000000_000332.mfc=/path/to/CORPUS-FILE1.mfc[0,332]
CORPUS-FILE1_000554_000599.mfc=/path/to/CORPUS-FILE1.mfc[554,599]
CORPUS-FILE1_000613_000770.mfc=/path/to/CORPUS-FILE1.mfc[613,770]
CORPUS-FILE1_000770_001027.mfc=/path/to/CORPUS-FILE1.mfc[770,1027]
