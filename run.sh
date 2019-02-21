embedding_dir=embedding_$1
data_path=$2
rm $embedding_dir -rf
mkdir $embedding_dir
python3 word2vec.py $data_path $embedding_dir

