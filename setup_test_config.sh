if [[ $- == *i* ]]
then
	echo Shell is properly in interactive mode
else
	echo Error: Shell needs to be invoked using interactive flag
	echo Without the -i, bash cannot access conda activate in subshell
	echo bash -i $0
	exit 1
fi




conda create -n test_enigma -y python=3.7 pip
conda activate test_enigma

#env_dir=$(conda info | grep "envs directories" | awk '{print $4}')
#env_dir=$env_dir/test_enigma

#conda install pip python=3.7 -y
#cd ~/src

#tmp = $(basename $(dirname $(pwd)))
#if [[ tmp -eq "enigma" ]]
#then
#	echo Do not run this from the enigma repo
#	echo It will download another version
#	exit(1)
#fi

git clone git@tako:enigma
pip install ./enigma
pip install -r enigma/requirements.txt

cd enigma
ln -s /data/test_data

pytest

