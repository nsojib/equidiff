cd ~
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .


mkdir envs


cd ~/mimicgen/envs
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
pip install -e .


cd ~/mimicgen/envs
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout d0b37cf214bd24fb590d182edb6384333f67b661
pip install -e .

pip install mujoco==2.3.2


cd ~/mimicgen/mimicgen/scripts
python download_datasets.py --dataset_type source --tasks all

python generate_training_configs_for_public_datasets.py


cd ~/mimicgen/envs
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
cd robosuite-task-zoo
git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed


mkdir ~/mimicgen/configs/
mkdir ~/mimicgen/configs/kitchen/
mkdir ~/mimicgen/configs/coffee/


