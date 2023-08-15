sudo apt-get install aria2

data_dir=~/TorchAL/data

# RSNA Pneumonia Detection Challenge (JPG files)
rsna_torrent=https://academictorrents.com/download/95588a735c9ae4d123f3ca408e56570409bcf2a9.torrent
rsna_old_folder_name=$data_dir/kaggle-pneumonia-jpg
rsna_new_folder_name=$data_dir/rsna
aria2c --seed-time=0 -d $data_dir $rsna_torrent
mv $rsna_old_folder_name $rsna_new_folder_name

# remove all torrent and aria2 file
rm "$data_dir"/*.torrent
rm "$data_dir"/*.aria2



