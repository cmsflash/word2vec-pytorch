all: 100

100:
	./run.sh 100 zhihu_100.txt

zhihu:
	./run.sh zhihu zhihu.txt

en:
	./run.sh en en.txt

zh:
	./run.sh zh zh.txt

char:
	./run.sh char char.txt

.PHONY: 100 zhihu en zh all char
