wnae_dir=$(echo $PWD | rev | cut -d/ -f2- | rev)
old_python_path=$PYTHONPATH
export PYTHONPATH=$wnae_dir
make clean
make html
export PYTHONPATH=$old_python_path
