#!/usr/bin/python
# rosanna milner
# ------------------------------------------------
import sys
import commands
import argparse
import numpy as np
import htkmfc	# for reading in HTK feature files

# ------------------------------------------------
def find_speech(scpname):
    	# reading scp
	scplines = []
	with open(scpname) as f:
		for line in f:
			fname = line.split("=")[1].split("[")[0]
			label = line.split("=")[1].split("[")[0].split("/")[len(line.split("=")[1].split("[")[0].split("/"))-1].split(".")[0]
			beg = int(line.split("[")[1].split(",")[0])
			end = int(line.split(",")[1].split("]")[0])
			dur = end - beg
			scplines.append([fname, label, beg, dur, end])
	# reading htk
	array_dict = {}
	count = 0
	for [fname, label, beg, dur, end] in scplines:
		features = htkmfc.HTKFeat_read(fname)
		array_dict[label+"_seg"+str(count)] = [np.array(features.getall()[beg:end]), beg, dur, end]
		count += 1
	return array_dict

# ------------------------------------------------
def clean(scores, clusters):
	# remove scores with already clustered clusters
	clean_scores = []
	for s in scores:
		spkr1 = s[0]
		spkr2 = s[4]
		if spkr1 in clusters:
			if spkr2 in clusters:
				clean_scores.append(s)
	return clean_scores

# ------------------------------------------------
def ahs_clust(covar1, covar2, cov_type, merge_len):
	# ARITHMETIC HARMONIC SPHERICITY
	# for diag covariances
	if cov_type == "diagonal":
		trace1 = 0.0
		trace2 = 0.0
		for i in range(len(covar1)):
			trace1 += covar2[i]/covar1[i]
			trace2 += covar1[i]/covar2[i]
	# for full covariance
	elif cov_type == "full":
		trace1 = np.trace(np.matrix(covar2) * np.matrix(covar1).I)
		trace2 = np.trace(np.matrix(covar1) * np.matrix(covar2).I)
	distance = np.log(trace1*trace2) - 2.0 * np.log(len(covar1))
	return distance

# ------------------------------------------------
def euclidean(len1, mean1, len2, mean2):
	# EUCLIDEAN DISTANCE
	const = ((len1*len2)/(len1+len2))
	sqsum = 0.0
	for i in range(len(mean1)):
		sqsum += np.power((mean1[i] - mean2[i]),2)
	distance = const * np.sqrt(sqsum)
	return distance

# ------------------------------------------------
def delta_bic(len1, cov1, len2, cov2, len_merge, cov_merge, lda, d, cov_type):
	# DELTA BIC 
	if cov_type == "diagonal":
		merged = len_merge * getlogdet_diag(cov_merge)
		sep1 = len1 * getlogdet_diag(cov1)
		sep2 = len2 * getlogdet_diag(cov2)
	elif cov_type == "full":
	# for full covariances
		(sign, logdet) = np.linalg.slogdet( cov_merge )
		merged = len_merge * logdet 
		sep1 = len1 * np.log( np.linalg.det( cov1 ) )
		sep2 = len2 * np.log( np.linalg.det( cov2 ) )
	pen = ( (d*(d+3)) / 4 ) * np.log( len_merge ) 
	# Tranter2006
	# deltaBIC = 0.2(Nlog|Sigma| - N1log|Sigma1| - N2log|Sigma2|) - lambda*pen
	# pen = ( (d*(d+3)) / 4 ) logN
	deltaBIC = (0.5 * (merged - sep1 - sep2)) - (lda * pen)
	return deltaBIC

# ------------------------------------------------
def getlogdet_diag(cov):
	# find log determinant
	sum = 0.0
	for n in cov:
		sum += np.log(n)
	return sum

# ------------------------------------------------
def updatemean(mean1, len1, mean2, len2):
	# find new mean from two
	new_mean = 2 * (mean1 + mean2) / (len1 + len2)
	return new_mean

# ------------------------------------------------
def updatecov(cov1, len1, mean1, cov2, len2, mean2):
	# find new length and covariance from two 
	new_cov = len1*cov1 + len2*cov2 + (mean1-mean2)*(mean1-mean2)*( len1 * len2 / (len1+len2) )
	return new_cov / (len1+len2)

# ------------------------------------------------
def getmean(data):
	d = data[0].shape[0]
	mean = np.zeros(d)
	for frame in data:
		mean += frame
	mean =  mean / len(data)
	return mean

# ------------------------------------------------
def getcov(data, cov_type):
	d = data[0].shape[0]
	if cov_type == "diagonal":
		mean = np.zeros(d)
		for frame in data:
			mean += frame
		mean = mean / len(data)
		cov = np.zeros(d)
		for frame in data:
			cov += (frame - mean) ** 2
		cov = cov / len(data)
	elif cov_type == "full":
		mean = np.zeros(d)
		mean2 = np.zeros((d, d))
		for frame in data:
			mean += frame
			mean2 += (np.matrix(frame).T*np.matrix(frame))
		mean =  mean / len(data)
		mean2 = mean2 / len(data)
		# covariance = E(XX.T) - mumu.T
		cov = (mean2 - (np.matrix(mean).T * np.matrix(mean)))
	return cov

# ------------------------------------------------
def distance_metric(metric, len1, mean1, cov1, len2, mean2, cov2, merge_len, merge_mean, merge_cov, lda, d, cov_type):
	if metric == "bic":
		score = delta_bic(len1, cov1, len2, cov2, merge_len, merge_cov, lda, d, cov_type)
	elif metric == "ahs":
		score = ahs_clust(cov1, cov2, cov_type, merge_len)
	elif metric == "euclid":
		score = euclidean(len1, mean1, len2, mean2)
	else:
		print("INVALID DISTANCE METRIC (must be BIC, AHS, EUCLID)")
		sys.exit(1)
	return score

# ------------------------------------------------
def clustering(f, data_array, args):
	# initialise
	score = -100000000
	tmplbl = data_array.keys()[0]
	d = len(data_array[tmplbl][0])
	if args.decision_metric == "bic":
		print("Penalty (lambda):\t%f" % args.penalty)
	print("Decision threshold:\t%f" % args.threshold)

	# fill stats list with covars and lengths
	clusters, scores, stats, short_segments = [], [], {}, []
	for label in data_array.keys():
		mean = getmean(data_array[label][0])
		covar = getcov(data_array[label][0], args.covar_type)
		if args.trace:
			print("Segments for initial clusters:")
			print(label, data_array[label][1:], data_array[label][0].shape, len(data_array[label][0]))
		stats[label] = [covar, len(data_array[label][0]), mean]
		if (data_array[label][2]/100.0) > args.short_time:
			clusters.append(label)
		else:
			short_segments.append(label)

	# cluster pairings loop
	start = True
	print("Initial clusters:\t%d\n" % len(clusters))
	while (len(clusters) != args.max_clusters):
		# do initial cluster pairings
		if start:
			for spkr1 in stats:
				[cov1, len1, mean1] = stats[spkr1]
				if spkr1 in clusters:
					for spkr2 in stats:
						[cov2, len2, mean2] = stats[spkr2]
						if spkr2 in clusters:
							if spkr1 != spkr2:
								merge_len = len1+len2
								merge_mean = updatemean(mean1, len1, mean2, len2)
								merge_cov = updatecov(cov1, len1, mean1, cov2, len2, mean2)
								# get score
								score = distance_metric(args.decision_metric, len1, mean1, cov1, len2, mean2, cov2, merge_len, merge_mean, merge_cov, args.penalty, d, args.covar_type)
								if args.max_clusters == 0:
									if score < args.threshold:
										scores.append([spkr1, cov1, len1, mean1, spkr2, cov2, len2, mean2, score])
								else:
									scores.append([spkr1, cov1, len1, mean1, spkr2, cov2, len2, mean2, score])
								if args.trace:
									print("[%s,%s] %f" % (spkr1, spkr2, score))
			start = False
		else:
			# non-initial loop, pair all clusters with new merged cluster only
			for spkr1 in stats:
				[cov1, len1, mean1] = stats[spkr1]
				if spkr1 in clusters:
					[spkr2, cov2, len2, mean2] = [new_spkr, new_cov, new_len, new_mean]
					if spkr1 != new_spkr:
						merge_len = len1 + new_len
						merge_mean = updatemean(mean1, len1, mean2, len2)
						merge_cov = updatecov(cov1, len1, mean1, cov2, len2, mean2)
						# get score
						score = distance_metric(args.decision_metric, len1, mean1, cov1, len2, mean2, cov2, merge_len, merge_mean, merge_cov, args.penalty, d, args.covar_type)
						if args.max_clusters == 0:
							if score < args.threshold:
								scores.append([spkr1, cov1, len1, mean1, spkr2, cov2, len2, mean2, score])
						else:
							scores.append([spkr1, cov1, len1, mean1, spkr2, cov2, len2, mean2, score])
						if args.trace:
							print("[%s,%s] %f" % (spkr1, spkr2, score))
		# sort scores to find lowest score
		ordered_scores = sorted(scores, key=lambda scores: scores[8])

		# if low score exists
		if len(ordered_scores) != 0:
			[spkr1, cov1, len1, mean1, spkr2, cov2, len2, mean2, score] = ordered_scores[0]
			# remove each from clusters list
			clusters.remove(spkr1)
			clusters.remove(spkr2)
			# remove scores containing above speakers from score list
			scores = clean(scores, clusters)
			# find new cluster details
			new_spkr = "%s+%s" % (spkr1, spkr2)
			clusters.append(new_spkr)
			new_len = len1+len2
			new_mean = updatemean(mean1, len1, mean2, len2)
			new_cov = updatecov(cov1, len1, mean1, cov2, len2, mean2)
			# add new cluster to stats
			stats[new_spkr] = [new_cov, new_len, new_mean]
			del stats[spkr1]
			del stats[spkr2]
			if args.trace:
				print("MERGED=[%s] %f" % (new_spkr, score))
		# else no score must exist below threshold
		else: 
			print("\nSTOP merging pairs as lowest %s score above threshold of %.2f..." % (args.decision_metric, args.threshold))
			print("#clusters:\t%d" % (len(clusters)))
			break
		# only one cluster left in clusters list
		if len(clusters) == 1:
			print("\nSTOP merging pairs as only one cluster left...")
			print("#clusters:\t%d" % (len(clusters)))
			break
		# else max_clusters reached
	else:
		print("\nSTOP merging pairs as reached max clusters...")
		print("#clusters:\t%d" % (len(clusters)))


	# short segments only
	# for each short segment check which found cluster it belongs too. new clusters are NOT updated for the newly added short segments as this could influence decisions dependent on order of matching short segment to cluster
	if short_segments != []:
		print("Num short segments:\t%d" % len(short_segments))
		new_clusters = {}
		for c in clusters:
			new_clusters[c] = []
		# short segment
		oldclusters = clusters
		for spkr1 in short_segments:
			[cov1, len1, mean1] = stats[spkr1]
			scores = []
			# compare with every found cluster
			for spkr2 in oldclusters:
				[cov2,len2,mean2] = stats[spkr2]
				score = distance_metric("euclid", len1, mean1, cov1, len2, mean2, cov2, "", "", "", "", "","")
				scores.append([spkr1, cov1, len1, mean1, spkr2, cov2, len2, mean2, score])
			ordered_scores = sorted(scores, key=lambda scores: scores[8])
			[spkr1, cov1, len1, mean1, spkr2, cov2, len2, mean2, score] = ordered_scores[0]
			new_clusters[spkr2].append(spkr1)
		clusters = []
		for spkr in new_clusters:
			clusters.append("+".join([spkr] + new_clusters[spkr]))

	# final result
	print("\nClusters found:\t%d\n" % len(clusters))
	return clusters

# ------------------------------------------------
def create_rttm(folder, name, clusters, data_array):
	commands.getoutput("mkdir %s" % folder)
	hypothesis = folder+"/"+name+".sys.rttm"
	print("Saving file here:\t%s" % hypothesis)
	count, rttmsegments = 0, {}
	for c in clusters:
		cname = "SPKR"+str(count)
		count += 1
		segments = c.split("+")
		for s in segments:
			[beg, dur, end] = data_array[s][1:]
			fname = s.split("_seg")[0]
			if fname not in rttmsegments:
				rttmsegments[fname] = []
			rttmsegments[fname].append([float(beg)/100.0, float(dur)/100.0, cname])
	# printing
	with open(hypothesis, 'w') as f: 
		for fname in rttmsegments:
			sort_segs = sorted(rttmsegments[fname], key=lambda s:s[0])
			for seg in sort_segs:
				print >> f, "SPEAKER %s 1 %f %f <NA> <NA> %s <NA>" % (fname, seg[0], seg[1], seg[2])
	return hypothesis

# ------------------------------------------------
def score(hypothesis, reference, collar):
	com = "./md-eval.pl -r %s -s %s -c %s" % (reference, hypothesis, collar)
	scores = commands.getoutput(com).split("\n")
	print("---------------------------------------------")
	for s in scores:
		print(s)

# ------------------------------------------------
# MAIN
# ------------------------------------------------

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("scp", help="SCP file containing speaker pure segments for clustering")
parser.add_argument("-d", "--decision-metric", help="Metric for decision and stop criterion", type=str, choices=["ahs", "bic", "euclid", "gish"], default="bic")
parser.add_argument("-c", "--covar-type", help="Covariance type", type=str, choices=["diagonal","full"], default="diagonal")
parser.add_argument("-p", "--penalty", help="Penalty (lambda) for BIC only", type=float, default=1.0)
parser.add_argument("-t", "--threshold", help="Decision threshold for stopping", type=float, default=0.0)
parser.add_argument("-m", "--max-clusters", help="Stop clustering when number of clusters reached", type=int, default=1)
parser.add_argument("-s", "--short-time", help="Any segments less than specified time (seconds) are not included in clustering process, but a decision is made the euclidean distance between the segment and each cluster found", type=float, default=0.0)
parser.add_argument("-f", "--folder", help="Folder to save output rttm", type=str, default="./")
parser.add_argument("-r", "--reference", help="Reference RTTM file for scoring", type=str)
parser.add_argument("--trace", help="Prints information during the clustering process", action="store_true")
parser.add_argument("--collar", help="Collar for scoring DER", type=float, default=0.25)
args = parser.parse_args()


# setup
print("---------------------------------------------")
print("Decision metric:\t%s" % args.decision_metric)
print("Covariance type:\t%s" % args.covar_type)
if args.max_clusters != 0:
	print("Max clusters:\t\t%d" % args.max_clusters)


# find files and read in mfc
name = args.scp.split("/")[len(args.scp.split("/"))-1].split(".")[0]
data_array = find_speech(args.scp)


# cluster
clusters = clustering(name, data_array, args)


# create rttm
print("---------------------------------------------")
lbl = "/cluster-%s_%s_p%.2f_t%.2f_m%d_s%.2f/" % (args.decision_metric, args.covar_type, args.penalty, args.threshold, args.max_clusters, args.short_time)
hypothesis = create_rttm(args.folder+lbl, name, clusters, data_array)


# score
if args.reference:
	score(hypothesis, args.reference, args.collar)
