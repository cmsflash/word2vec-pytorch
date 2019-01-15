all: 100

100:
	python3 word2vec.py zhihu_100.txt embedding_100.txt

zhihu:
	python3 word2vec.py zhihu.txt embedding_zhihu.txt

en:
	python3 word2vec.py en.txt embedding_en.txt

zh:
	python3 word2vec_reproduction.py zh.txt embedding_zh.txt chinese-297.txt

.PHONY: 100 zhihu en zh all
