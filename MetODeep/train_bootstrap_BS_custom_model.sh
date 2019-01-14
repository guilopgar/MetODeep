#!/bin/bash


# ----- Arguments management -----

while [[ $# -gt 0 ]]
do

key="$1"
case $key in
    -l|--learning-rate)
    LR=$2
    shift # past argument
    shift # past value
    ;;
    -t|--transfer-leayer)
    TL=$2
    shift # past argument
    shift # past value
    ;;
    -n|--nclass)
    N=$2
    shift # past argument
    shift # past value
    ;;
    -b|--background)
    B=$2
    shift # past argument
    shift # past value
    ;;
    -p|--phospho-residue)
    PH=$2
    shift # past argument
    shift # past value
    ;;
    -f|--fine-output)
    FT=$2
    shift # past argument
    shift # past value
    ;;
    -r|--result-output)
    RES=$2
    shift # past argument
    shift # past value
    ;;
    -h|--help)
	echo "Usage: $0 [OPTION]...
Example: $0 -l 0.003 -t 5

Arguments:
 -l, --learning-rate=NUM	Learning rate value used when fine-tuning
 				the model. It should be a FLOAT value.
 				Default value is 0.001
 -t, --transfer-leayer=NUM	Transfer-leayer value used when fine-tuning 
 				the model. It should be an INTEGER value.
 				Default value is 1
 -n, --nclass=NUM		Number of class value used when fine-tuning 
 				the model. It should be an INTEGER value.
 				Default value is 1
 -p, --phospho-residue=STRING	'ST' or 'STY' phosphorylatable residues used
 				during pre-training. Default value is STY
 -b, --background=STRING	Prefix of the existing pre-trained model 
 				using general phosphorylation data (without the ST or STY suffix).
 				Default value is all-phos-data/models/pre-train/custom_general_(value of -p)
 -f, --fine-output=DIRECTORY 	Directory in which fine-tuned models will be
 				temporally stored (it will be created if it does not exist). 
 				An indermediate directory whose name indicates the learning rate,
 				transfer-leayer and nclass used values will be created.
 				Default value is all-phos-data/models/fine-tune
 -r, --result-output=DIRECTORY 	Directory in which results files will be
 				stored (it will be created if it does not exist). 
 				An indermediate directory whose name indicates the learning rate,
 				transfer-leayer and nclass used values will be created. 
 				Default value is all-phos-data/results
 -h, --help			It shows this help and exits"
	exit
	;;
    *)    # unknown option
	echo "ERROR: $1 is not a valid option. Use -h (--help) to see correct usage"
    exit
    ;;
esac
done

lr=${LR:-0.001}
tl=${TL:-1}
n=${N:-1}
p=${PH:-STY}
b=${B:-all-phos-data/models/pre-train/custom_general_}
b=${b}$p
ft=${FT:-all-phos-data/models/fine-tune}
res=${RES:-all-phos-data/results}


# ----- Bootstrap strategy ----- 

# Directories to create
c=bs-$p-nclass$n-lr$lr-tl$tl
ft=${ft}/$c
res=${res}/$c

# Create if not exist
mkdir -p $ft
mkdir -p $res

# Store current directory
d=$PWD

for i in {1..100}
do
	# Train
	# Using bootstrap strategy
	# Using the custom general model pre-trained on both training and test S, T, Y residues
	# Using nclass=1, transfer-leayer=1 and lr=0.001 by default
	python ../MusiteDeep_Keras2.0/MusiteDeep/train_kinase.py -input fasta_files_bs/all_train_MetOx_bs_$i.fasta -background-prefix $b -output-prefix $ft/metox_model_custom_general_${p}_bs_$i -residue-types M -nclass $n -transferlayer $tl -lr $lr

	# Test
	python ../MusiteDeep_Keras2.0/MusiteDeep/predict.py -input fasta_files_bs/all_test_MetOx_bs_$i.fasta -predict-type custom -model-prefix $ft/metox_model_custom_general_${p}_bs_$i -output $res/metox_result_custom_general_${p}_bs_$i

	# Remove model files
	cd $ft
	ls | grep metox_model_custom_general_${p}_bs_.*_HDF5 | xargs -d"\n" rm
	cd $d
done

# Remove all parameters files except the last one
cd $ft
ls | grep -v metox_model_custom_general_${p}_bs_..._ | xargs -d"\n" rm
