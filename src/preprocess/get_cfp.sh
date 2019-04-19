# prepare CFP data first
dst_dir=../../data/CFP
list_file_name=/home/ubuntu/zms/data/cfp/cfp-dataset/Data/list_name.txt
dst_file_name=align_img_list.txt
python pre_cfp_data.py $dst_dir $list_file_name $dst_file_name

