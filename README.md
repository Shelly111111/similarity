# similarity
基于paddlepaddle的中文相似度匹配

	首先，执行train.py进行训练，生成model，会存储在bow_pairwise/final该文件夹下。
	再次，执行test.py对model，进行测试，确认model的分类结果是否理想，如不尽人意可重复执行上述动作，训练会以final文件夹下的model继续训练。
	最后，Similarity.py是对我们输入的句子与已有句子进行相似度匹配。已有句子保存在data/data12739/infer文件中。

	PS:datacollect.py是对data12739中的文本问题进行收集所执行的代码段，已执行过一次无需再次执行。